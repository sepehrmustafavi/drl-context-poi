"""Raw, lazy inference and cost/latency profiling; never reads subset caches."""

from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .cache import bits
from .model import pack, ensemble_predict
from .policy import QNetwork, drop_for
from .predictor import load_ensemble
from .util import load_torch, file_hash, write_json, environment


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_cost(csv, predictor, out, count=100, device=None):
    store, models, dev = load_ensemble(csv, predictor, device)
    rs = store.split("fit", True)
    # Use training observations only, spread deterministically through the fit interval.
    chosen = [rs[i] for i in np.linspace(0, len(rs) - 1, min(count, len(rs)), dtype=int)]
    times = [[] for _ in store.sources]
    for j in range(len(store.sources)):
        candidates = [r for r in chosen if store.descriptors(r)[0][j]]
        # Warm up relevant code paths independently of timed observations.
        for r in candidates[: min(10, len(candidates))]:
            m = np.zeros((1, len(store.sources)), bool)
            m[0, j] = True
            ensemble_predict(models, pack(store, [r], m, dev), store.cfg["score_chunk"])
        for r in candidates:
            m = np.zeros((1, len(store.sources)), bool)
            m[0, j] = True
            synchronize(dev)
            start = perf_counter()
            batch = pack(store, [r], m, dev)
            ensemble_predict(models, batch, store.cfg["score_chunk"])
            synchronize(dev)
            times[j].append((perf_counter() - start) * 1000)
    if not any(times):
        raise ValueError("No source is available in training requests")
    pooled = float(np.median([x for ts in times for x in ts]))
    medians = [float(np.median(ts)) if ts else pooled for ts in times]
    costs = (np.array(medians) / pooled).tolist()
    write_json(
        out,
        dict(
            costs=costs,
            raw_median_ms=medians,
            counts=list(map(len, times)),
            sources=store.sources,
            unavailable_source_fallback="pooled training median; unavailable sources are action-masked",
            normalization_ms=pooled,
            reference_cost=sum(costs),
            environment=environment(),
            data_identity=store.identity,
            predictor_hashes=[file_hash(Path(predictor) / f"member_{i}.pt") for i in range(len(models))],
            definition="retrieval + features + bounded block encoding + ensemble prediction refresh; batch=1",
            caveat="additive proxy; actual multi-step latency is measured separately",
        ),
    )


class Inference:
    def __init__(self, csv, predictor, cfg, policy=None):
        self.store, self.models, self.dev = load_ensemble(csv, predictor, cfg["device"])
        self.cfg = cfg
        self.j = len(self.store.sources)
        self.s = 2**self.j
        self.costs = np.asarray(cfg["costs"], np.float32)
        self.budget = float(cfg["budget_fraction"] * self.costs.sum())
        self.net = None
        self.method = None
        if policy:
            ck = load_torch(policy, self.dev)
            hashes = [file_hash(Path(predictor) / f"member_{i}.pt") for i in range(len(self.models))]
            if ck["predictor_identity"] != hashes or ck["data_identity"] != self.store.identity:
                raise ValueError("Wrong policy checkpoint")
            for k in ("costs", "lambda_cost", "budget_fraction", "max_acquisitions", "disabled_sources"):
                if ck["config"][k] != cfg[k]:
                    raise ValueError("Retrain policy when changing " + k)
            self.net = QNetwork(ck["input_dim"], ck["config"]["q_hidden"], ck["actions"]).to(self.dev)
            self.net.load_state_dict(ck["model"])
            self.net.eval()
            self.method = ck["method"]
        self.rng = np.random.default_rng(cfg["seed"])

    @torch.no_grad()
    def run(self, r, method=None, fixed=0):
        method = method or self.method
        if not method:
            raise ValueError("Specify a policy")
        st = self.store
        j = self.j
        mask = 0
        traces = []
        blocks = {}
        available, desc = st.descriptors(r)
        for source in self.cfg.get("disabled_sources", []):
            available[st.sources.index(source)] = False
            desc[st.sources.index(source)] = 0
        cap = self.cfg["max_acquisitions"]

        def valid(m):
            selected = bits(m, j)
            ac = (~selected) & available & (self.costs <= self.budget - self.costs[selected].sum() + 1e-6)
            if selected.sum() >= cap:
                ac[:] = False
            return ac

        feasible = [
            m
            for m in range(self.s)
            if np.all(~bits(m, j) | available)
            and bits(m, j).sum() <= cap
            and self.costs[bits(m, j)].sum() <= self.budget + 1e-6
        ]
        while True:
            selected = bits(mask, j)
            batch = pack(st, [r], selected[None], self.dev, blocks)
            p, z, h, u = ensemble_predict(self.models, batch, st.cfg["score_chunk"])
            rem = max(0, self.budget - float(self.costs[selected].sum()))
            latent = z[0].cpu().numpy()
            entropy = float(h.item()) / max(np.log(len(st.catalog)), 1)
            dis = float(u.item()) / max(np.log(len(st.catalog)), 1)
            if drop_for(method) == "uncertainty":
                entropy = dis = 0
            if drop_for(method) == "representation":
                latent[:] = 0
            state = np.r_[
                latent,
                selected,
                available,
                desc,
                self.costs,
                entropy,
                dis,
                rem / self.budget if self.budget else 0,
                selected.sum() / max(1, cap),
            ].astype(np.float32)
            ac = valid(mask)
            if method in ("base", "all", "fixed", "direct", "bandit"):
                if traces:  # Subset selected in one decision; now return its refreshed prediction.
                    return p[0].cpu().numpy(), mask, traces + [j]
                if method == "base":
                    desired = 0
                elif method == "all":
                    desired = sum(1 << a for a in range(j) if available[a])
                elif method == "fixed":
                    desired = fixed & sum(1 << a for a in range(j) if available[a])
                    if desired not in feasible:
                        desired = 0
                else:
                    q = self.net(torch.tensor(state, device=self.dev)).cpu().numpy()
                    desired = max(feasible, key=lambda m: (q[m], -m))
                if desired == 0:
                    return p[0].cpu().numpy(), 0, [j]
                mask = desired
                traces = [a for a in range(j) if (mask >> a) & 1]
                continue
            valid_actions = np.r_[ac, True]
            if method == "no_stop" and ac.any():
                valid_actions[-1] = False
            if method == "random":
                action = int(self.rng.choice(np.flatnonzero(valid_actions)))
            elif method == "entropy":
                choices = [a for a in self.cfg["fixed_order"] if ac[a]]
                action = choices[0] if choices and entropy > self.cfg["entropy_threshold"] else j
            else:
                if self.net is None:
                    raise ValueError("Learned policy requires --policy checkpoint")
                q = self.net(torch.tensor(state, device=self.dev)).cpu().numpy()
                action = int(np.argmax(np.where(valid_actions, q, -np.inf)))
            traces.append(action)
            if action == j:
                return p[0].cpu().numpy(), mask, traces
            mask |= 1 << action


def latency(csv, predictor, cfg, out, policy=None, method=None, count=1000, warmup=50, repeats=3, fixed=0):
    engine = Inference(csv, predictor, cfg, policy)
    rs = engine.store.split("test")
    rs = [rs[i] for i in np.linspace(0, len(rs) - 1, min(count, len(rs)), dtype=int)]
    if not rs:
        raise ValueError("Empty test requests")
    for i in range(warmup):
        engine.run(rs[i % len(rs)], method, fixed)
    measurements = []
    if engine.dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(engine.dev)
    for rep in range(repeats):
        times = []
        for r in rs:
            synchronize(engine.dev)
            t = perf_counter()
            engine.run(r, method, fixed)
            synchronize(engine.dev)
            times.append((perf_counter() - t) * 1000)
        measurements.append(
            dict(
                repetition=rep,
                p50_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, 0.95)),
                mean_ms=float(np.mean(times)),
                serial_requests_per_second=1000 / float(np.mean(times)),
            )
        )
    params = sum(sum(p.numel() for p in m.parameters()) for m in engine.models)
    if engine.net:
        params += sum(p.numel() for p in engine.net.parameters())
    write_json(
        out,
        dict(
            method=method or engine.method,
            requests=len(rs),
            warmup=warmup,
            repetitions=measurements,
            parameters=params,
            environment=environment(),
            config=cfg,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.dev)
            if engine.dev.type == "cuda"
            else None,
            path="raw retrieval; no subset cache; descriptor/base/policy/refresh included",
            concurrency="run on an otherwise idle device; repetitions are not training seeds",
        ),
    )
