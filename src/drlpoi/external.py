"""Common split/catalog exchange for independently audited upstream recommenders.

This module does NOT rename a simplified model as GETNext, DCHL or iPCM.
Official implementations run in isolated environments and export full-catalog scores.
"""

from pathlib import Path
import json
import numpy as np
from .data import Store
from .metrics import rank_of, per_request, summarize
from .util import digest, write_json, read_json

UPSTREAM = {
    "getnext": "https://github.com/songyangco/GETNext",
    "dchl": "https://github.com/icmpnorequest/SIGIR2024_DCHL",
    "ipcm": "https://github.com/songchaocn/iPCM",
}


def export(csv, cfg, out):
    st = Store(csv, cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "catalog.json", st.catalog)
    st.df.to_csv(out / "events.csv", index_label="event_index")
    st.df[st.t < st.cuts[0]].to_csv(out / "predictor_fit_events.csv", index_label="event_index")
    ids = {}
    for split in ("fit", "policy", "val", "test"):
        rs = st.split(split)
        ids[split] = [r.id for r in rs]
        with open(out / f"{split}_requests.jsonl", "w") as f:
            for r in rs:
                # Targets are stored separately: feature adapters must read only *_features files.
                f.write(
                    json.dumps(
                        dict(
                            request_id=r.id,
                            user=r.user,
                            prefix_event_indices=list(r.prefix),
                            query_time=float(st.t[r.prefix[-1]]),
                            target_index=st.target(r),
                        )
                    )
                    + "\n"
                )
        with open(out / f"{split}_features.jsonl", "w") as f:
            for r in rs:
                f.write(
                    json.dumps(
                        dict(
                            request_id=r.id,
                            user=r.user,
                            prefix_event_indices=list(r.prefix),
                            query_time=float(st.t[r.prefix[-1]]),
                        )
                    )
                    + "\n"
                )
    manifest = dict(
        data_identity_sha256=digest(st.identity),
        catalog_sha256=digest(st.catalog),
        request_ids_sha256={s: digest(v) for s, v in ids.items()},
        audit=st.audit(),
        upstream_repositories=UPSTREAM,
        contract_version=1,
    )
    write_json(out / "manifest.json", manifest)
    return manifest


def import_scores(csv, cfg, scores, manifest, split="test"):
    st = Store(csv, cfg)
    rs = st.split(split)
    byid = {r.id: r for r in rs}
    meta = read_json(manifest)
    expected = dict(
        data_identity_sha256=digest(st.identity),
        catalog_sha256=digest(st.catalog),
        request_ids_sha256=digest([r.id for r in rs]),
    )
    for k, v in expected.items():
        if meta.get(k) != v:
            raise ValueError(f"External provenance mismatch: {k}")
    for k in ("source_commit", "model_name", "seed", "training_config"):
        if k not in meta:
            raise ValueError(f"Missing external provenance: {k}")
    if (
        meta.get("train_split") != "fit"
        or meta.get("target_time_used") is not False
        or meta.get("temporal_leakage_audited") is not True
    ):
        raise ValueError("Baseline requires fit-only training and an explicit temporal leakage audit")
    seen = set()
    rows = []
    with open(scores) as f:
        for line in f:
            x = json.loads(line)
            rid = x["request_id"]
            if rid not in byid or rid in seen:
                raise ValueError("Unknown or duplicate request ID")
            seen.add(rid)
            r = byid[rid]
            logits = np.asarray(x["scores"], np.float64)
            if logits.shape != (len(st.catalog),) or not np.isfinite(logits).all():
                raise ValueError("Expected finite scores over the exact full catalog")
            target = st.target(r)
            nll = None
            if target >= 0:
                peak = logits.max()
                nll = float(np.log(np.exp(logits - peak).sum()) + peak - logits[target])
            rank = rank_of(logits, target)
            rows.append(
                dict(
                    request_id=rid,
                    user=r.user,
                    in_catalog=target >= 0,
                    cost=0.0,
                    acquisitions=0,
                    **per_request(rank, nll),
                )
            )
    if seen != set(byid):
        raise ValueError("External model omitted requests; do not silently drop cold-start users/POIs")
    return rows, summarize(rows)
