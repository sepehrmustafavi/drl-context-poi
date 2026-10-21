"""Common-protocol adapters using pinned upstream model classes.

No upstream train.py/run.py is executed: several use incompatible splits or test
checkpoint selection. Graphs, metadata and clusters here use the fit interval only.
"""

from pathlib import Path
from types import SimpleNamespace
from collections import defaultdict
import importlib.util
import subprocess
import numpy as np
import torch
from torch import nn
from .data import Store
from .metrics import rank_of, per_request, summarize
from .util import (
    read_json,
    write_json,
    file_hash,
    digest,
    seed_everything,
    device_of,
    save_torch,
    load_torch,
    environment,
)

ROOT = Path(__file__).resolve().parents[2]


def fetch(name, root=None):
    meta = read_json(ROOT / "configs/upstream.json")[name]
    path = Path(root or ROOT / "artifacts/upstream") / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", meta["repository"], str(path)], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(path),
                "sparse-checkout",
                "set",
                "--no-cone",
                "/model.py",
                "/python/model.py",
                "/LICENSE",
                "/README.md",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", meta["commit"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != meta["commit"]:
        raise ValueError("Upstream commit differs from pin: " + name)
    module = path / meta.get("module", "model.py")
    expected = subprocess.check_output(["git", "-C", str(path), "show", f"HEAD:{module.relative_to(path)}"])
    if module.read_bytes() != expected:
        raise ValueError("Upstream model has uncommitted modifications")
    return path, dict(repository=meta["repository"], commit=actual, model_sha256=file_hash(module))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location("drlpoi_upstream_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sparse(matrix, device):
    m = matrix.tocoo()
    return torch.sparse_coo_tensor(
        np.stack([m.row, m.col]), m.data.astype(np.float32), m.shape, device=device
    ).coalesce()


def normalize(matrix):
    from scipy import sparse as sp

    return sp.diags(1 / np.maximum(np.asarray(matrix.sum(1)).ravel(), 1e-12)) @ matrix


def time_clusters(st, classes=20):
    """The upstream contiguous temporal clustering DP, fit-only, safe empty bins."""
    n = 72
    cats = max(1, len(st.cmap) + 1)
    counts = np.zeros((n, cats))
    ids = np.flatnonzero(st.t < st.cuts[0])
    slices = np.minimum(71, (st.t[ids] % 86400 / 1200).astype(int))
    np.add.at(counts, (slices, st.cat[ids]), 1)
    counts /= np.maximum(counts.sum(1, keepdims=True), 1)
    cs = np.vstack([np.zeros(cats), np.cumsum(counts, 0)])
    sq = np.r_[0, np.cumsum((counts * counts).sum(1))]
    loss = np.full((n + 1, classes + 1), np.inf)
    back = np.zeros((n + 1, classes + 1), int)
    loss[0, 0] = 0
    for k in range(1, classes + 1):
        for end in range(k, n + 1):
            starts = np.arange(k - 1, end)
            costs = sq[end] - sq[starts] - ((cs[end] - cs[starts]) ** 2).sum(1) / (end - starts)
            values = loss[starts, k - 1] + costs
            best = np.argmin(values)
            loss[end, k], back[end, k] = values[best], starts[best]
    bins = np.zeros(n, int)
    end = n
    for k in range(classes, 0, -1):
        start = back[end, k]
        bins[start:end] = k - 1
        end = start
    return bins


class Adapted(nn.Module):
    def __init__(self, st, cfg, name, module, device):
        super().__init__()
        from scipy import sparse as sp

        self.st, self.cfg, self.name, self.dev = st, cfg, name, device
        n, d = len(st.catalog), cfg["dim"]
        fitids = np.flatnonzero(st.t < st.cuts[0])
        self.users = {u: i for i, u in enumerate(sorted(st.df.iloc[fitids].user_id.unique()))}
        self.unknown = len(self.users)
        histories = defaultdict(list)
        for i in fitids:
            histories[self.users[st.df.user_id.iat[i]]].append(i)
        self.histories = histories
        coords, cats, freq = [], [], []
        for p in st.catalog:
            ids = st.df.index[(st.df.poi_id == p) & (st.t < st.cuts[0])]
            coords.append(st.coords[ids[0]])
            cats.append(st.cat[ids[0]])
            freq.append(len(ids))
        self.categories = np.asarray(cats)
        self.coords = np.asarray(coords)
        self.register_buffer("category_ids", torch.tensor(cats, device=device))
        edges = [(st.p[a] - 1, st.p[b] - 1) for ids in histories.values() for a, b in zip(ids, ids[1:])]
        er, ec = zip(*edges) if edges else ([], [])
        adjacency = sp.coo_matrix((np.ones(len(er)), (er, ec)), shape=(n, n)).tocsr()
        if name == "sasrec":
            args = SimpleNamespace(
                device=str(device),
                hidden_units=d,
                maxlen=cfg["recent"],
                dropout_rate=cfg["dropout"],
                num_blocks=cfg["layers"],
                num_heads=cfg["heads"],
                norm_first=True,
            )
            # Extra input-only UNK embedding; output catalog is strictly 1..N.
            self.negatives = {
                u: np.setdiff1d(np.arange(n), np.unique(st.p[ids] - 1)) for u, ids in histories.items()
            }
            self.model = module.SASRec(len(self.users), n + 1, args)
            with torch.no_grad():
                self.model.item_emb.weight[n + 1].zero_()
        elif name == "getnext":
            features = np.c_[np.asarray(freq), np.eye(len(st.cmap) + 1)[cats], self.coords]
            self.register_buffer("node_features", torch.tensor(features, dtype=torch.float32, device=device))
            self.register_buffer("adj", sparse(normalize(adjacency + sp.eye(n)), device))
            self.adjacency = adjacency
            self.gcn = module.GCN(features.shape[1], [d], d, cfg["dropout"])
            self.attention = module.NodeAttnMap(features.shape[1], d)
            self.user = module.UserEmbeddings(len(self.users) + 1, d)
            self.category = module.CategoryEmbeddings(len(st.cmap) + 1, d)
            self.time = module.Time2Vec("sin", d)
            self.fuse1, self.fuse2 = module.FuseEmbeddings(d, d), module.FuseEmbeddings(d, d)
            self.seq = module.TransformerModel(
                n, len(st.cmap) + 1, 4 * d, cfg["heads"], 4 * d, cfg["layers"], cfg["dropout"]
            )
        elif name == "dchl":
            from sklearn.neighbors import BallTree

            up_rows, up_cols = [], []
            directed = set()
            for u, ids in histories.items():
                pois = [int(st.p[i] - 1) for i in ids]
                up_rows.extend([u] * len(set(pois)))
                up_cols.extend(sorted(set(pois)))
                following = set()
                for p in reversed(pois):
                    directed.update((p, q) for q in following)
                    following.add(p)
            up = sp.coo_matrix(
                (np.ones(len(up_rows)), (up_rows, up_cols)), shape=(len(self.users) + 1, n)
            ).tocsr()
            pu = normalize(up.T)
            up = normalize(up).tolil()
            up[-1] = up[:-1].mean(axis=0)  # unknown-user fallback, no future data
            dr, dc = zip(*sorted(directed)) if directed else ([], [])
            di = sp.coo_matrix((np.ones(len(dr)), (dr, dc)), shape=(n, n)).tocsr()
            tree = BallTree(np.radians(self.coords), metric="haversine")
            neighbors = tree.query_radius(np.radians(self.coords), r=cfg.get("geo_radius_km", 2.5) / 6371)
            rows = np.repeat(np.arange(n), [len(x) for x in neighbors])
            geo = sp.coo_matrix((np.ones(len(rows)), (rows, np.concatenate(neighbors))), shape=(n, n))
            for key, mat in dict(
                HG_up=up.tocsr(),
                HG_pu=pu,
                HG_poi_src=normalize(di),
                HG_poi_tar=normalize(di.T),
                poi_geo_graph=normalize(geo),
            ).items():
                self.register_buffer(key, sparse(mat, device))
            self.pad_all_train_sessions = None  # unused by official forward
            args = SimpleNamespace(
                emb_dim=d,
                temperature=0.1,
                num_mv_layers=3,
                num_geo_layers=3,
                num_di_layers=3,
                dropout=cfg["dropout"],
            )
            self.model = module.DCHL(len(self.users) + 1, n, args, device)
            from types import MethodType

            self.model.cal_loss_infonce = MethodType(chunked_infonce, self.model)
        elif name == "ipcm":
            from sklearn.cluster import KMeans

            nr = min(cfg.get("region_clusters", 100), n)
            self.regions = KMeans(
                n_clusters=nr, max_iter=1000, n_init=10, random_state=cfg["seed"]
            ).fit_predict(self.coords[:, ::-1])
            self.time_bins = time_clusters(st)
            self.time_values = np.array(
                [
                    (
                        np.flatnonzero(self.time_bins == k).min()
                        + np.flatnonzero(self.time_bins == k).max()
                        + 1
                    )
                    / 144
                    for k in range(20)
                ]
            )
            self.register_buffer(
                "regions_tensor", torch.tensor(self.regions, dtype=torch.long, device=device)
            )
            self.poi = module.POIGraph(n, d)
            self.region = module.POIGraph(nr, d)
            self.user = module.UserEmbeddings(len(self.users) + 1, d)
            self.time = module.Time2Vec(d)
            self.fuse = module.FuseEmbeddings(2 * d, 2 * d)
            self.seq = module.TransformerModel(
                n, nr, 20, 4 * d, cfg["heads"], 4 * d, cfg["layers"], cfg["dropout"]
            )
            self.register_buffer("edge", torch.tensor(np.array([er, ec]), dtype=torch.long, device=device))
            redge = (
                np.array([[self.regions[a], self.regions[b]] for a, b in edges], dtype=np.int64)
                .reshape(-1, 2)
                .T
            )
            self.register_buffer("region_edge", torch.tensor(redge, device=device))
            ur, uc, uv = [], [], []
            self.visits = defaultdict(set)
            for u, ids in histories.items():
                counts = np.bincount(st.p[ids] - 1, minlength=n)
                nonzero = np.flatnonzero(counts)
                ur.extend([u] * len(nonzero))
                uc.extend(nonzero)
                uv.extend(counts[nonzero] / counts.sum())
                for i in ids:
                    self.visits[(u, self.time_id(i))].add(int(st.p[i] - 1))
            mat = sp.coo_matrix((uv, (ur, uc)), shape=(len(self.users) + 1, n)).tolil()
            mat[-1] = mat[:-1].mean(axis=0)
            self.register_buffer("user_poi", sparse(mat.tocsr(), device))
        else:
            raise ValueError(name)
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        # Official DCHL calls functional dropout without a training flag.
        if self.name == "dchl":
            self.model.di_hconv_network.dropout = self.cfg["dropout"] if mode else 0.0
        return self

    def time_id(self, idx):
        return int(self.time_bins[min(71, int(self.st.t[idx] % 86400 / 1200))])

    def forward(self, requests):
        st, dev = self.st, self.dev
        n = len(st.catalog)
        users = torch.tensor([self.users.get(r.user, self.unknown) for r in requests], device=dev)
        zero = torch.zeros((), device=dev)
        if self.name == "sasrec":
            seq = np.zeros((len(requests), self.cfg["recent"]), np.int64)
            for i, r in enumerate(requests):
                ids = r.prefix[-self.cfg["recent"] :]
                seq[i, -len(ids) :] = st.p[ids]
            z = self.model.log2feats(seq)[:, -1]
            return z @ self.model.item_emb.weight[1 : n + 1].T, {}, zero
        if self.name == "dchl":
            logits, lu, lp = self.model(self, {"user_idx": users})
            return logits, {}, 0.1 * (lu + lp)
        if self.name == "getnext":
            pois = self.gcn(self.node_features, self.adj)
            # Compute only requested flow-map rows, algebraically identical to NodeAttnMap.
            wh = self.node_features @ self.attention.W
            left = wh @ self.attention.a[: self.attention.out_features]
            right = wh @ self.attention.a[self.attention.out_features :]
            ue = self.user(users)
            ue = torch.where(
                (users == self.unknown)[:, None], self.user.user_embedding.weight[: self.unknown].mean(0), ue
            )
        else:
            pois = self.poi(torch.arange(n, device=dev), self.edge)
            regions = self.region(torch.arange(len(set(self.regions)), device=dev), self.region_edge)
            all_users = self.user(
                torch.arange(len(self.users) + 1, device=dev), torch.sparse.mm(self.user_poi, pois)
            )
            ue = torch.where(
                (users == self.unknown)[:, None], all_users[: self.unknown].mean(0), all_users[users]
            )
        sequences, lengths, last_pois = [], [], []
        for row, r in enumerate(requests):
            ids = r.prefix[-self.cfg["recent"] :]
            values = []
            for i in ids:
                p = int(st.p[i] - 1)
                known = p < n
                pe = pois[p] if known else pois.mean(0)
                if self.name == "getnext":
                    ce = self.category(torch.tensor(int(st.cat[i]), device=dev))
                    te = self.time(
                        torch.tensor([[st.t[i] % 86400 / 86400]], device=dev, dtype=torch.float32)
                    )[0]
                    values.append(torch.cat([self.fuse1(ue[row], pe), self.fuse2(te, ce)]))
                else:
                    re = regions[self.regions[p]] if known else regions.mean(0)
                    te = self.time(
                        torch.tensor([[self.time_values[self.time_id(i)]]], device=dev, dtype=torch.float32)
                    )[0]
                    values.append(self.fuse(torch.cat([ue[row], pe]), torch.cat([te, re])))
            sequences.append(torch.stack(values))
            lengths.append(len(ids))
            last_pois.append(int(st.p[ids[-1]] - 1))
        x = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        causal = torch.triu(torch.ones(x.shape[1], x.shape[1], device=dev, dtype=torch.bool), 1)
        if self.name == "getnext":
            # Each sequence independently avoids padding affecting official transformer.
            outputs = [self.seq(v[:, None], causal[: len(v), : len(v)]) for v in sequences]
            logits = torch.stack([v[0][-1, 0] for v in outputs])
            time = torch.stack([v[1][-1, 0, 0] for v in outputs])
            cat = torch.stack([v[2][-1, 0] for v in outputs])
            additions = []
            for p in last_pois:
                if p >= n:
                    additions.append(torch.zeros(n, device=dev))
                else:
                    flow = (
                        torch.tensor(self.adjacency.getrow(p).toarray()[0], device=dev, dtype=torch.float32)
                        + 1
                    )
                    additions.append(self.attention.leakyrelu(left[p] + right.T).flatten() * flow)
            return logits + torch.stack(additions), dict(time=time, category=cat), zero
        padding = torch.arange(x.shape[1], device=dev)[None] >= torch.tensor(lengths, device=dev)[:, None]
        po, ti, re = self.seq(x, causal, padding)
        ix = torch.arange(len(requests), device=dev)
        last = torch.tensor(lengths, device=dev) - 1
        return po[ix, last], dict(time=ti[ix, last], region=re[ix, last], users=users), zero

    def loss(self, requests):
        logits, aux, extra = self(requests)
        targets = torch.tensor([self.st.target(r) for r in requests], device=self.dev)
        if self.name == "sasrec":
            rows = torch.arange(len(requests), device=self.dev)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[rows, targets], torch.ones_like(targets, dtype=torch.float32)
            )
            if logits.shape[1] > 1:
                negatives = []
                for r in requests:
                    choices = self.negatives[self.users[r.user]]
                    if not len(choices):
                        choices = np.delete(np.arange(logits.shape[1]), self.st.target(r))
                    negatives.append(int(choices[torch.randint(len(choices), (1,)).item()]))
                negative = torch.tensor(negatives, device=self.dev)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    logits[rows, negative], torch.zeros_like(targets, dtype=torch.float32)
                )
            return loss
        loss = nn.functional.cross_entropy(logits, targets) + extra
        if self.name == "getnext":
            cats = torch.tensor([self.st.cat[r.target] for r in requests], device=self.dev)
            times = torch.tensor(
                [self.st.t[r.target] % 86400 / 86400 for r in requests], device=self.dev, dtype=torch.float32
            )
            loss = (
                loss
                + nn.functional.cross_entropy(aux["category"], cats)
                + 10 * nn.functional.mse_loss(aux["time"], times)
            )
        if self.name == "ipcm":
            region = self.regions_tensor[targets]
            times = torch.tensor([self.time_id(r.target) for r in requests], device=self.dev)
            loss = (
                loss
                + nn.functional.cross_entropy(aux["region"], region)
                + nn.functional.cross_entropy(aux["time"], times)
            )
        return loss

    @torch.no_grad()
    def probabilities(self, requests):
        logits, aux, _ = self(requests)
        p = logits.softmax(-1)
        if self.name == "ipcm":
            # Official ranking adjustment uses predicted time/region, never target time/location.
            for row in range(len(requests)):
                times = aux["time"][row].topk(min(5, aux["time"].shape[-1])).indices.tolist()
                regions = aux["region"][row].topk(min(20, aux["region"].shape[-1])).indices.tolist()
                u = int(aux["users"][row])
                visited = set().union(*(self.visits[(u, t)] for t in times))
                good = [k for k in visited if self.regions[k] in regions]
                p[row, good] += 1
            p /= p.sum(-1, keepdim=True)  # normalized adjusted scores define reported NLL
        return p


def baseline(csv, cfg, name, out, upstream_root=None, resume=False):
    """Fit only on fit, select on validation, then score EVERY test request."""
    from .cli import dump_results

    seed_everything(cfg["seed"], cfg["threads"])
    st, dev = Store(csv, cfg), device_of(cfg["device"])
    source, provenance = fetch(name, upstream_root)
    meta = read_json(ROOT / "configs/upstream.json")[name]
    module = load_module(name, source / meta.get("module", "model.py"))
    net = Adapted(st, cfg, name, module, dev)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = digest(dict(config=cfg, data=st.identity, source=provenance))
    bestpath = out / "model.pt"
    if bestpath.exists() and not resume:
        raise FileExistsError("Use --resume or a fresh baseline directory")
    opt = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    start, best, stale = 0, -1.0, 0
    if resume and (out / "last.pt").exists():
        ck = load_torch(out / "last.pt", dev)
        if ck["signature"] != signature:
            raise ValueError("Baseline resume config/data/source mismatch")
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        start, best, stale = ck["epoch"] + 1, ck["best"], ck["stale"]
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if dev.type == "cuda":
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
    train = st.split("fit", True)
    validation = st.split("val")
    if not train or not validation or not st.split("test"):
        raise ValueError("Empty fit/validation/test requests")
    for epoch in range(start, cfg["epochs"]):
        if stale >= cfg["patience"]:
            break
        net.train()
        losses = []
        order = rng.permutation(len(train))
        for a in range(0, len(train), cfg["batch_size"]):
            rs = [train[i] for i in order[a : a + cfg["batch_size"]]]
            opt.zero_grad(set_to_none=True)
            loss = net.loss(rs)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite baseline loss")
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        scores = []
        for a in range(0, len(validation), cfg["batch_size"]):
            rs = validation[a : a + cfg["batch_size"]]
            probs = net.probabilities(rs).cpu().numpy()
            scores.extend(per_request(rank_of(p, st.target(r)))["ndcg@10"] for r, p in zip(rs, probs))
        metric = float(np.mean(scores))
        if metric > best:
            best, stale = metric, 0
            save_torch(
                bestpath, dict(model=net.state_dict(), signature=signature, epoch=epoch, val_ndcg=best)
            )
        else:
            stale += 1
        save_torch(
            out / "last.pt",
            dict(
                model=net.state_dict(),
                signature=signature,
                epoch=epoch,
                optimizer=opt.state_dict(),
                best=best,
                stale=stale,
                rng=rng.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if dev.type == "cuda" else None,
            ),
        )
        print(dict(baseline=name, epoch=epoch, loss=float(np.mean(losses)), val_ndcg=metric), flush=True)
    ck = load_torch(bestpath, dev)
    net.load_state_dict(ck["model"])
    net.eval()
    rows = []
    requests = st.split("test")
    for a in range(0, len(requests), cfg["batch_size"]):
        rs = requests[a : a + cfg["batch_size"]]
        probs = net.probabilities(rs).cpu().numpy()
        for r, p in zip(rs, probs):
            y = st.target(r)
            rows.append(
                dict(
                    request_id=r.id,
                    user=r.user,
                    in_catalog=y >= 0,
                    cost=0.0,
                    acquisitions=0,
                    **per_request(rank_of(p, y), -np.log(max(float(p[y]), 1e-30)) if y >= 0 else None),
                )
            )
    summary = summarize(rows)
    summary.update(
        method=name,
        seed=cfg["seed"],
        config=cfg,
        source=provenance,
        environment=environment(),
        protocol="common chronological protocol adaptation; not original published split/results",
        data_identity=st.identity,
        validation_ndcg=ck["val_ndcg"],
        parameters=sum(p.numel() for p in net.parameters()),
    )
    dump_results(out / "test", rows, summary)
    write_json(out / "audit.json", st.audit())
    return summary


def chunked_infonce(model, emb1, emb2):
    """Exact upstream InfoNCE with stable logsumexp and bounded saved activations."""
    from torch.utils.checkpoint import checkpoint

    if not model.training:
        return emb1.new_zeros(())  # auxiliary loss is not used in inference
    loss = emb1.new_zeros(())
    # Capture slices in separate closures so checkpoint recomputation is stable.
    for start in range(0, len(emb1), 128):
        end = min(start + 128, len(emb1))

        def compute(a, b, lo=start, hi=end):
            positive = (a * b[lo:hi]).sum(-1) / model.ssl_temp
            return (torch.logsumexp(a @ b.T / model.ssl_temp, dim=-1) - positive).sum()

        loss = loss + checkpoint(compute, emb1[start:end], emb2, use_reentrant=False)
    return loss / len(emb1)


def baseline_latency(csv, cfg, name, checkpoint, out, upstream_root=None, count=1000, warmup=50, repeats=3):
    from time import perf_counter
    from .runtime import synchronize
    import psutil

    seed_everything(cfg["seed"], cfg["threads"])
    st, dev = Store(csv, cfg), device_of(cfg["device"])
    path, provenance = fetch(name, upstream_root)
    meta = read_json(ROOT / "configs/upstream.json")[name]
    net = Adapted(st, cfg, name, load_module(name, path / meta.get("module", "model.py")), dev)
    ck = load_torch(checkpoint, dev)
    if ck["signature"] != digest(dict(config=cfg, data=st.identity, source=provenance)):
        raise ValueError("Baseline timing configuration/source differs from training")
    net.load_state_dict(ck["model"])
    net.eval()
    requests = st.split("test")
    rs = [requests[i] for i in np.linspace(0, len(requests) - 1, min(count, len(requests)), dtype=int)]
    if not rs or count < 1 or repeats < 1 or warmup < 0:
        raise ValueError("Invalid timing sample")
    measurements = []
    for i in range(warmup):
        net.probabilities([rs[i % len(rs)]])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for rep in range(repeats):
        times = []
        for r in rs:
            synchronize(dev)
            start = perf_counter()
            net.probabilities([r])
            synchronize(dev)
            times.append((perf_counter() - start) * 1000)
        measurements.append(
            dict(
                repetition=rep,
                p50_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, 0.95)),
                per_request_ms=times,
            )
        )
    write_json(
        out,
        dict(
            method=name,
            requests=len(rs),
            warmup=warmup,
            repetitions=measurements,
            parameters=sum(p.numel() for p in net.parameters()),
            process_rss_bytes=psutil.Process().memory_info().rss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
            environment=environment(),
            source=provenance,
            checkpoint_sha256=file_hash(checkpoint),
            path="raw baseline forward including graph refresh, prefix encoding, adjustment and full-catalog scoring",
        ),
    )
