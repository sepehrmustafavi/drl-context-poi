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
        scores = prior.copy()
        if personal:
            local = Counter(st.p[list(r.prefix)])
            # Fixed tiny global tie-breaker; repeated personal visits are valid.
            scores = np.array([local[i + 1] for i in range(len(st.catalog))]) + prior / prior.sum() * 1e-6
        p = (scores + 1e-8) / (scores + 1e-8).sum()
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
