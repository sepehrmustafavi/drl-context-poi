from collections import Counter
import numpy as np
from .data import Store
from .metrics import per_request, rank_of, summarize


def popularity(csv, cfg, personal=False, split="test"):
    st = Store(csv, cfg)
    counts = Counter(st.p[st.t < st.cuts[0]])
    prior = np.array([counts[i + 1] + 1e-8 for i in range(len(st.catalog))], np.float64)
    rows = []
    for r in st.split(split):
        p = frequency_scores(st, r, prior, personal)
        y = st.target(r)
        rows.append(
            dict(
                request_id=r.id,
                user=r.user,
                in_catalog=y >= 0,
                cost=0.0,
                acquisitions=0,
                **per_request(rank_of(p, y), -np.log(p[y]) if y >= 0 else None),
            )
        )
    return rows, summarize(rows)


def frequency_scores(st, r, prior, personal):
    scores = prior.copy()
    if personal:
        local = Counter(st.p[list(r.prefix)])
        scores = np.array([local[i + 1] for i in range(len(st.catalog))]) + prior / prior.sum() * 1e-6
    return (scores + 1e-8) / (scores + 1e-8).sum()


def frequency_latency(csv, cfg, out, personal=False, count=1000, warmup=50, repeats=3):
    from time import perf_counter
    import psutil
    from .util import environment, write_json, seed_everything

    seed_everything(cfg["seed"], cfg["threads"])
    st = Store(csv, cfg)
    counts = Counter(st.p[st.t < st.cuts[0]])
    prior = np.array([counts[i + 1] + 1e-8 for i in range(len(st.catalog))], np.float64)
    requests = st.split("test")
    rs = [requests[i] for i in np.linspace(0, len(requests) - 1, min(count, len(requests)), dtype=int)]
    if not rs or min(count, repeats) < 1 or warmup < 0:
        raise ValueError("Invalid timing sample")
    for i in range(warmup):
        frequency_scores(st, rs[i % len(rs)], prior, personal)
    measurements = []
    for rep in range(repeats):
        times = []
        for r in rs:
            start = perf_counter()
            frequency_scores(st, r, prior, personal)
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
            method="personal" if personal else "popularity",
            requests=len(rs),
            warmup=warmup,
            repetitions=measurements,
            parameters=0,
            process_rss_bytes=psutil.Process().memory_info().rss,
            environment=environment(),
            path="raw CPU frequency scoring on reference host; fit counts/index initialized before timing",
        ),
    )
