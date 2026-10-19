"""Predeclared subgroup analysis and multiplicity correction."""

import json
from collections import defaultdict
import numpy as np
from .data import Store
from .util import write_json


def holm(pvalues):
    """Holm family-wise adjusted p-values; input order is retained."""
    p = np.asarray(pvalues, float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    prev = 0.0
    for i, idx in enumerate(order):
        prev = max(prev, min(1.0, (len(p) - i) * p[idx]))
        adjusted[idx] = prev
    return adjusted.tolist()


def subgroups(csv, cfg, results, out):
    st = Store(csv, cfg)
    fit_history = [len(r.prefix) for r in st.split("fit")]
    boundaries = np.quantile(fit_history, [1 / 3, 2 / 3])
    requests = {r.id: r for r in st.split("test")}
    groups = defaultdict(list)
    with open(results) as f:
        for line in f:
            row = json.loads(line)
            r = requests[row["request_id"]]
            group = ("short", "medium", "long")[int(np.searchsorted(boundaries, len(r.prefix), side="right"))]
            groups[group].append(row)
    report = {
        g: dict(
            requests=len(rows),
            users=len({r["user"] for r in rows}),
            ndcg10=float(np.mean([r["ndcg@10"] for r in rows])),
            cost=float(np.mean([r["cost"] for r in rows])),
        )
        for g, rows in groups.items()
    }
    write_json(
        out, dict(history_boundaries=boundaries.tolist(), boundary_fit="predictor-fit only", groups=report)
    )
