"""Raw, lazy inference and cost/latency profiling; never reads subset caches."""

from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from .cache import bits
from .model import RequestPrediction
from .policy import QNetwork, drop_for, LEARNED
from .predictor import load_ensemble
from .util import load_torch, file_hash, write_json, environment, seed_everything


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_cost(csv, predictor, out, count=100, device=None):
    store, models, dev = load_ensemble(csv, predictor, device)
    seed_everything(store.cfg["seed"], store.cfg["threads"])
    rs = store.split("fit", True)
    # Use training observations only, spread deterministically through the fit interval.
    chosen = [rs[i] for i in np.linspace(0, len(rs) - 1, min(count, len(rs)), dtype=int)]
    times = [[] for _ in store.sources]
    for j in range(len(store.sources)):
        candidates = [r for r in chosen if store.descriptors(r)[0][j]]
        # Warm up relevant code paths independently of timed observations.
        for r in candidates[: min(10, len(candidates))]:
            with torch.no_grad():
                request = RequestPrediction(store, models, r, dev)
                request.predict(1 << j)
        for r in candidates:
            with torch.no_grad():
                request = RequestPrediction(store, models, r, dev)
                request.predict(0)
                synchronize(dev)
                start = perf_counter()
                request.predict(1 << j)
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
        seed_everything(cfg["seed"], cfg["threads"])
        self.store, self.models, self.dev = load_ensemble(csv, predictor, cfg["device"])
        self.cfg = cfg
        self.j = len(self.store.sources)
        self.s = 2**self.j
        if cfg["sources"] != self.store.sources or cfg["dim"] != self.store.cfg["dim"]:
            raise ValueError("Predictor source order or representation dimension mismatch")
        self.costs = np.asarray(cfg["costs"], np.float32)
        if self.costs.shape != (self.j,) or not np.isfinite(self.costs).all() or (self.costs <= 0).any():
            raise ValueError("A positive finite cost is required for each source")
        self.budget = float(cfg["budget_fraction"] * self.costs.sum())
        if self.budget < 0 or cfg["max_acquisitions"] < 0:
            raise ValueError("Budget and acquisition cap must be nonnegative")
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
        if method in LEARNED and method != self.method:
            raise ValueError("Learned method must match its policy checkpoint")
        st = self.store
        j = self.j
        mask = 0
        traces = []
        request = RequestPrediction(st, self.models, r, self.dev)
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
            p, z, h, u = request.predict(mask)
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
    if min(count, repeats) < 1 or warmup < 0:
        raise ValueError("Positive timing count/repeats and nonnegative warmup required")
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
        engine.rng = np.random.default_rng(cfg["seed"])
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
                per_request_ms=times,
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
            request_ids=[r.id for r in rs],
            warmup=warmup,
            repetitions=measurements,
            parameters=params,
            predictor_parameters=sum(sum(p.numel() for p in m.parameters()) for m in engine.models),
            policy_parameters=sum(p.numel() for p in engine.net.parameters()) if engine.net else 0,
            process_rss_bytes=__import__("psutil").Process().memory_info().rss,
            acquisition_costs=engine.costs.tolist(),
            predictor_hashes=[
                file_hash(Path(predictor) / f"member_{i}.pt") for i in range(len(engine.models))
            ],
            environment=environment(),
            config=cfg,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.dev)
            if engine.dev.type == "cuda"
            else None,
            path="raw retrieval; no subset cache; descriptor/base/policy/refresh included",
            concurrency="run on an otherwise idle device; repetitions are not training seeds",
        ),
    )


def complexity(csv, predictor, cfg, out, count=1000, warmup=50, repeats=3):
    """Raw shape/workload scaling on trained weights, with a controlled fixed policy.

    This measures computational scaling, not recommendation quality of truncated
    catalogs or modified history lengths. Separate latency jobs measure deployment.
    """
    import psutil

    seed = cfg["seed"]
    engine = Inference(csv, predictor, cfg)
    full_models = list(engine.models)
    catalog = len(engine.store.catalog)
    history = engine.store.cfg["recent"]
    maximum = len(engine.store.sources)
    engine.cfg = dict(cfg, budget_fraction=1.0, max_acquisitions=maximum)
    engine.costs = np.ones(maximum, dtype=np.float32)
    engine.budget = float(maximum)
    original = engine.store.cfg
    requests = engine.store.split("test")
    if not requests or min(count, repeats) < 1 or warmup < 0:
        raise ValueError("Invalid scaling timing sample")
    rs = [requests[i] for i in np.linspace(0, len(requests) - 1, min(count, len(requests)), dtype=int)]
    reference = dict(catalog=catalog, history=history, ensemble=len(full_models), acquisitions=maximum)
    settings = []
    for factor, values in dict(
        catalog=sorted({max(1, catalog // 4), max(1, catalog // 2), catalog}),
        history=sorted({max(1, history // 4), max(1, history // 2), history}),
        ensemble=list(range(1, len(full_models) + 1)),
        acquisitions=list(range(maximum + 1)),
    ).items():
        settings.extend((factor, dict(reference, **{factor: value})) for value in values)
    results = []
    try:
        for factor, setting in settings:
            engine.models = full_models[: setting["ensemble"]]
            engine.store.cfg = dict(original, recent=setting["history"])
            for model in full_models:
                model.npoi = setting["catalog"]
            fixed = (1 << setting["acquisitions"]) - 1
            for i in range(warmup):
                engine.run(rs[i % len(rs)], "fixed", fixed)
            if engine.dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats(engine.dev)
            reps = []
            actual = []
            for rep in range(repeats):
                times = []
                for r in rs:
                    synchronize(engine.dev)
                    t = perf_counter()
                    _, mask, _ = engine.run(r, "fixed", fixed)
                    synchronize(engine.dev)
                    times.append((perf_counter() - t) * 1000)
                    if rep == 0:
                        actual.append(mask.bit_count())
                reps.append(
                    dict(
                        repetition=rep, p50_ms=float(np.median(times)), p95_ms=float(np.quantile(times, 0.95))
                    )
                )
            results.append(
                dict(
                    factor=factor,
                    setting=setting,
                    repetitions=reps,
                    actual_mean_acquisitions=float(np.mean(actual)),
                    parameters_resident=sum(sum(p.numel() for p in m.parameters()) for m in full_models),
                    rss_bytes=psutil.Process().memory_info().rss,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.dev)
                    if engine.dev.type == "cuda"
                    else None,
                )
            )
    finally:
        engine.models = full_models
        engine.store.cfg = original
        for model in full_models:
            model.npoi = catalog
    write_json(
        out,
        dict(
            seed=seed,
            requests=len(rs),
            warmup=warmup,
            repeats=repeats,
            results=results,
            reference=reference,
            environment=environment(),
            scope="controlled raw fixed-policy workload; not accuracy or a retrained deployment policy; unavailable blocks reduce actual acquisitions",
            memory_scope="all checkpoint weights remain resident; M sweep varies active compute, not allocated parameter memory",
            complexity="Transformer base O(M*(L^2*d+L*d^2)); source encoding O(M*A*block*d^2); repeated full scoring O(M*(A+1)*N*d); Q O((A+1)*(state*q+q^2+q*J))",
        ),
    )
