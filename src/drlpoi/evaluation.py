"""Predeclared paired families and descriptive quality/cost diagnostics."""

from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from .analysis import holm
from .metrics import paired_user_bootstrap
from .util import read_json, write_json, digest


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate request IDs: " + str(path))
    return rows


def average_seeds(paths):
    sets = [{r["request_id"]: r for r in read_rows(p)} for p in paths]
    if not sets or any(set(s) != set(sets[0]) for s in sets):
        raise ValueError("Seed runs must have identical request IDs")
    rows = []
    for rid, x in sets[0].items():
        if any(s[rid]["user"] != x["user"] or s[rid]["in_catalog"] != x["in_catalog"] for s in sets):
            raise ValueError("Seed request provenance differs")
        row = dict(x)
        row["ndcg@10"] = float(np.mean([s[rid]["ndcg@10"] for s in sets]))
        rows.append(row)
    return rows


def compare_family(path, out, samples=2000):
    spec = read_json(path)
    if not spec.get("comparisons") or not spec.get("predeclared"):
        raise ValueError("Use a nonempty family declared before inspecting test outcomes")
    results = []
    for item in spec["comparisons"]:
        a, b = average_seeds(item["a"]), average_seeds(item["b"])
        result = paired_user_bootstrap(a, b, samples=samples)
        results.append(dict(name=item["name"], a=item["a"], b=item["b"], **result))
    adjusted = holm([x["paired_cluster_signflip_p"] for x in results])
    for row, p in zip(results, adjusted):
        row["holm_adjusted_p"] = p
    write_json(
        out,
        dict(
            family=spec["name"],
            family_sha256=digest(spec),
            comparisons=results,
            interpretation="users resampled after pairing and averaging training seeds per request; report seed SD separately",
        ),
    )


def diagnostics(root, out, tolerance=0.002):
    """Descriptive, post-evaluation analysis; never selects deployment settings."""
    root = Path(root)
    records = []
    groups = defaultdict(list)
    for path in sorted(root.rglob("summary.json")):
        s = read_json(path)
        if s.get("seed") != 11 or s.get("method") == "oracle" or not s.get("all"):
            continue
        relative = path.relative_to(root).parts
        if relative[0] not in ("nyc", "tky") or relative[1] != "transformer":
            continue
        record = dict(
            scenario=str(path.parent),
            ndcg=s["all"]["request_macro"]["ndcg@10"],
            cost=s["cost_mean"],
            acquisitions=s.get("acquisitions_mean"),
        )
        timing = path.parent / "latency.json"
        record["latency"] = read_json(timing)["repetitions"] if timing.exists() else None
        # Avoid copying per-request timing vectors into the descriptive report.
        if record["latency"]:
            record["latency"] = [
                {k: v for k, v in rep.items() if k != "per_request_ms"} for rep in record["latency"]
            ]
        records.append(record)
        groups[relative[0]].append(record)
    matched = []
    frontier = {}
    for ds, items in groups.items():
        frontier[ds] = [
            r
            for r in items
            if not any(
                x["cost"] <= r["cost"]
                and x["ndcg"] >= r["ndcg"]
                and (x["cost"] < r["cost"] or x["ndcg"] > r["ndcg"])
                for x in items
            )
        ]
        for i, a in enumerate(items):
            for b in items[i + 1 :]:
                if abs(a["ndcg"] - b["ndcg"]) <= tolerance:
                    matched.append(
                        dict(
                            dataset=ds,
                            a=a["scenario"],
                            b=b["scenario"],
                            ndcg_difference=a["ndcg"] - b["ndcg"],
                            cost_difference=a["cost"] - b["cost"],
                        )
                    )
    traces = []
    acquisition_groups = {}
    for ds in ("nyc", "tky"):
        path = root / ds / "transformer/seed11/policies/ddqn/test/requests.jsonl"
        if not path.exists():
            continue
        rows = read_rows(path)
        basepath = root / ds / "transformer/seed11/references/base/test/requests.jsonl"
        base = {r["request_id"]: r for r in read_rows(basepath)} if basepath.exists() else {}
        bins = defaultdict(list)
        for row in rows:
            bins[row["acquisitions"]].append(row)
        acquisition_groups[ds] = {
            str(k): dict(
                requests=len(v),
                ndcg=float(np.mean([r["ndcg@10"] for r in v])),
                mean_gain_over_base=float(
                    np.mean([r["ndcg@10"] - base[r["request_id"]]["ndcg@10"] for r in v])
                )
                if base
                else None,
                cost=float(np.mean([r["cost"] for r in v])),
            )
            for k, v in bins.items()
        }
        for success in (True, False):
            chosen = sorted(
                (r for r in rows if bool(r["ndcg@10"]) == success),
                key=lambda r: (-r["acquisitions"], r["request_id"]),
            )[:3]
            traces.extend(
                dict(
                    dataset=ds,
                    success_at10=success,
                    selection="most acquisitions, then request ID",
                    **r,
                    base_ndcg=base.get(r["request_id"], {}).get("ndcg@10"),
                )
                for r in chosen
            )
    write_json(
        out,
        dict(
            seed=11,
            quality_tolerance=tolerance,
            descriptive_only=True,
            scenarios=records,
            pareto_frontier=frontier,
            matched_quality_pairs=matched,
            traces=traces,
            acquisition_groups=acquisition_groups,
        ),
    )
