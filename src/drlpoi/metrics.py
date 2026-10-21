from collections import defaultdict
import numpy as np


def rank_of(probabilities, target):
    if target < 0:
        return None
    p = np.asarray(probabilities)
    if p.ndim != 1 or not np.isfinite(p).all() or target >= len(p):
        raise ValueError("Ranking needs finite full-catalog scores and a valid target")
    # Exact full-catalog rank with deterministic catalog-ID tie breaking.
    return int(1 + np.sum(p > p[target]) + np.sum((p == p[target]) & (np.arange(len(p)) < target)))


def per_request(rank, nll=None):
    out = {f"recall@{k}": float(rank is not None and rank <= k) for k in (5, 10, 20)}
    out["ndcg@10"] = float(1 / np.log2(rank + 1)) if rank and rank <= 10 else 0.0
    out["mrr"] = 1 / rank if rank else 0.0
    if nll is not None:
        out["nll"] = float(nll)
    return out


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation requests")
    names = ["recall@5", "recall@10", "recall@20", "ndcg@10", "mrr"]
    eligible = [r for r in rows if r["in_catalog"]]
    out = {"requests": len(rows), "in_catalog": len(eligible), "out_of_catalog": len(rows) - len(eligible)}
    out["in_catalog_requests"] = len(eligible)
    for label, group in [("all", rows), ("in_catalog", eligible)]:
        if not group:
            out[label] = None
            continue
        users = defaultdict(list)
        for row in group:
            users[row["user"]].append(row)
        out[label] = {
            "request_macro": {k: float(np.mean([r[k] for r in group])) for k in names},
            "user_macro": {
                k: float(np.mean([np.mean([r[k] for r in rs]) for rs in users.values()])) for k in names
            },
        }
    out["nll_in_catalog"] = float(np.mean([r["nll"] for r in eligible])) if eligible else None
    nll_users = defaultdict(list)
    for r in eligible:
        nll_users[r["user"]].append(r["nll"])
    out["nll_user_macro_in_catalog"] = (
        float(np.mean([np.mean(v) for v in nll_users.values()])) if eligible else None
    )
    out["cost_mean"] = float(np.mean([r["cost"] for r in rows]))
    out["acquisitions_mean"] = float(np.mean([r["acquisitions"] for r in rows]))
    return out


def paired_user_bootstrap(a, b, key="ndcg@10", samples=2000, seed=11):
    if samples < 1 or not a or not b:
        raise ValueError("Positive bootstrap count and nonempty paired requests required")
    if len({r["request_id"] for r in a}) != len(a) or len({r["request_id"] for r in b}) != len(b):
        raise ValueError("Duplicate paired request IDs")
    a, b = {r["request_id"]: r for r in a}, {r["request_id"]: r for r in b}
    if set(a) != set(b):
        raise ValueError("Paired comparison needs exactly identical request IDs")
    grouped = defaultdict(list)
    for rid, x in a.items():
        y = b[rid]
        if x["user"] != y["user"] or x["in_catalog"] != y["in_catalog"]:
            raise ValueError("Request provenance mismatch")
        grouped[x["user"]].append(x[key] - y[key])
    groups = list(grouped.values())
    sums = np.array([sum(g) for g in groups])
    counts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples)
    for i in range(samples):
        ix = rng.integers(len(groups), size=len(groups))
        deltas[i] = sums[ix].sum() / counts[ix].sum()
    observed = float(sums.sum() / counts.sum())
    null = np.array(
        [(sums * rng.choice([-1.0, 1.0], size=len(groups))).sum() / counts.sum() for _ in range(samples)]
    )
    pvalue = float((1 + np.sum(np.abs(null) >= abs(observed))) / (samples + 1))
    return dict(
        metric=key,
        paired_cluster_signflip_p=pvalue,
        delta=float(sums.sum() / counts.sum()),
        ci95=np.quantile(deltas, [0.025, 0.975]).tolist(),
        resampling_unit="user; pooled request-macro difference",
        samples=samples,
        seed=seed,
    )
