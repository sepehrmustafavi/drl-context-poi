"""Subset lattice cache. Label-dependent rewards are kept out of observation arrays."""

from pathlib import Path
import numpy as np
from .model import pack, ensemble_predict
from .metrics import rank_of
from .predictor import load_ensemble
from .util import digest, file_hash, read_json, write_json


def bits(mask, j):
    return np.array([(mask >> i) & 1 for i in range(j)], dtype=bool)


def build(csv, predictor, out, split, device=None):
    store, models, dev = load_ensemble(csv, predictor, device)
    rs = store.split(split, eligible=split == "policy")
    if not rs:
        raise ValueError(f"No {split} requests")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    j = len(store.sources)
    s = 2**j
    d = store.cfg["dim"]
    model_hashes = [file_hash(Path(predictor) / f"member_{i}.pt") for i in range(len(models))]
    identity = dict(
        data=store.identity, model_hashes=model_hashes, split=split, request_ids=[r.id for r in rs]
    )
    signature = digest(identity)
    meta_path = out / "metadata.json"
    if meta_path.exists() and read_json(meta_path)["signature"] != signature:
        raise ValueError("Cache directory contains another model/data/split")
    metadata = dict(
        signature=signature,
        identity=identity,
        dim=d,
        sources=store.sources,
        requests=len(rs),
        catalog_size=len(store.catalog),
        complete=False,
    )
    write_json(meta_path, metadata)
    shapes = dict(
        latent=(len(rs), s, d),
        entropy=(len(rs), s),
        disagreement=(len(rs), s),
        available=(len(rs), j),
        descriptors=(len(rs), j),
        reward_nll=(len(rs), s),
        rank=(len(rs), s),
        done=(len(rs),),
    )
    arrays = {}
    for key, shape in shapes.items():
        path = out / f"{key}.npy"
        dtype = np.uint8 if key in ("available", "done") else np.int32 if key == "rank" else np.float32
        arrays[key] = np.lib.format.open_memmap(
            path, mode="r+" if path.exists() else "w+", dtype=dtype, shape=shape
        )
    masks = np.stack([bits(m, j) for m in range(s)])
    request_info = [
        dict(
            id=r.id,
            user=r.user,
            in_catalog=store.target(r) >= 0,
            history=len(r.prefix),
            target_index=store.target(r),
        )
        for r in rs
    ]
    write_json(out / "requests.json", request_info)
    for idx, r in enumerate(rs):
        if arrays["done"][idx]:
            continue
        av, desc = store.descriptors(r)
        arrays["available"][idx] = av
        arrays["descriptors"][idx] = desc
        # Bounded subset chunks limit peak GPU memory. This cache is never used for latency claims.
        for start in range(0, s, store.cfg["batch_size"]):
            selected = masks[start : start + store.cfg["batch_size"]] & av
            batch = pack(store, [r] * len(selected), selected, dev)
            prob, z, h, u = ensemble_predict(models, batch, store.cfg["score_chunk"])
            prob = prob.cpu().numpy()
            target = store.target(r)
            sl = slice(start, start + len(selected))
            arrays["latent"][idx, sl] = z.cpu().numpy()
            arrays["entropy"][idx, sl] = h.cpu().numpy()
            arrays["disagreement"][idx, sl] = u.cpu().numpy()
            arrays["rank"][idx, sl] = [rank_of(p, target) or -1 for p in prob]
            arrays["reward_nll"][idx, sl] = [
                -np.log(max(float(p[target]), 1e-30)) if target >= 0 else 0 for p in prob
            ]
        # Flush state/reward arrays before committing this resumable record.
        for key, a in arrays.items():
            if key != "done":
                a.flush()
        arrays["done"][idx] = 1
        arrays["done"].flush()
        if idx % 1000 == 0:
            print(f"cache {split}: {idx}/{len(rs)}", flush=True)
    metadata["complete"] = True
    write_json(meta_path, metadata)


class Cache:
    def __init__(self, path, cfg):
        self.path = Path(path)
        self.meta = read_json(self.path / "metadata.json")
        self.cfg = cfg
        if not self.meta["complete"]:
            raise ValueError("Incomplete cache; resume cache command")
        if cfg["sources"] != self.meta["sources"] or cfg["dim"] != self.meta["dim"]:
            raise ValueError("Cache source order or representation dimension mismatch")
        if (
            cfg.get("cost_profile_data_identity", self.meta["identity"]["data"])
            != self.meta["identity"]["data"]
        ):
            raise ValueError("Cost profile belongs to another cached data/index configuration")
        self.requests = read_json(self.path / "requests.json")
        self.j = len(self.meta["sources"])
        self.s = 2**self.j
        for key in ("latent", "entropy", "disagreement", "available", "descriptors", "reward_nll", "rank"):
            setattr(self, key, np.load(self.path / f"{key}.npy", mmap_mode="r"))
        costs = cfg["costs"]
        if costs is None or len(costs) != self.j or any(c <= 0 or not np.isfinite(c) for c in costs):
            raise ValueError("Positive profiled costs are required; run profile-cost and use --cost-profile")
        self.available = np.array(self.available)
        self.descriptors = np.array(self.descriptors)
        for source in cfg.get("disabled_sources", []):
            self.available[:, self.meta["sources"].index(source)] = 0
            self.descriptors[:, self.meta["sources"].index(source)] = 0
        self.costs = np.asarray(costs, np.float32)
        self.budget = float(cfg["budget_fraction"] * self.costs.sum())
        if self.budget < 0 or cfg["max_acquisitions"] < 0:
            raise ValueError("Budget and acquisition cap must be nonnegative")
        self.state_dim = self.meta["dim"] + 4 * self.j + 4
        self._feasible = {}
        self._availability_codes = np.asarray(self.available, dtype=np.int32) @ (1 << np.arange(self.j))
        for av in range(self.s):
            self._feasible[av] = tuple(
                m
                for m in range(self.s)
                if (m & ~av) == 0
                and m.bit_count() <= cfg["max_acquisitions"]
                and self.costs[bits(m, self.j)].sum() <= self.budget + 1e-6
            )

    def valid(self, i, mask, force_stop=False):
        selected = bits(mask, self.j)
        remaining = self.budget - float(self.costs[selected].sum())
        acquire = (~selected) & self.available[i].astype(bool) & (self.costs <= remaining + 1e-6)
        if selected.sum() >= self.cfg["max_acquisitions"]:
            acquire[:] = False
        stop = True
        if force_stop and acquire.any():
            stop = False
        return np.r_[acquire, stop]

    def feasible(self, i):
        return self._feasible[int(self._availability_codes[i])]

    def state(self, i, mask, drop=None):
        selected = bits(mask, self.j)
        k = int(selected.sum())
        remaining = max(0, self.budget - float(self.costs[selected].sum()))
        entropy = float(self.entropy[i, mask]) / max(np.log(self.meta["catalog_size"]), 1)
        u = float(self.disagreement[i, mask]) / max(np.log(self.meta["catalog_size"]), 1)
        latent = np.array(self.latent[i, mask])
        if drop == "uncertainty":
            entropy = u = 0.0
        if drop == "representation":
            latent[:] = 0
        return np.r_[
            latent,
            selected,
            self.available[i],
            self.descriptors[i],
            self.costs,
            entropy,
            u,
            remaining / self.budget if self.budget else 0,
            k / max(1, self.cfg["max_acquisitions"]),
        ].astype(np.float32)

    def objective(self, i, mask):
        return -float(self.reward_nll[i, mask]) - self.cfg["lambda_cost"] * float(
            self.costs[bits(mask, self.j)].sum()
        )

    def compatible(self, other):
        a, b = self.meta["identity"], other.meta["identity"]
        if a["data"] != b["data"] or a["model_hashes"] != b["model_hashes"]:
            raise ValueError("Caches must use identical predictor checkpoints and data")
