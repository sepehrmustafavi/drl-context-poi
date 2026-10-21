"""Aggregate only completed result files; never invent missing runs."""

from collections import defaultdict
from pathlib import Path
import csv
import numpy as np
from .util import read_json, write_json


def report(root, out):
    root = Path(root)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    groups = defaultdict(list)
    for path in sorted(root.rglob("summary.json")):
        summary = read_json(path)
        if not summary.get("all"):
            continue
        row = dict(
            path=str(path),
            method=summary.get("method", "external_or_raw"),
            seed=summary.get("seed"),
            population=summary.get("evaluation_population", "all_requests"),
            **summary["all"]["request_macro"],
            cost=summary["cost_mean"],
            requests=summary["requests"],
        )
        for population in ("all", "in_catalog"):
            for weighting, values in (summary.get(population) or {}).items():
                if isinstance(values, dict):
                    row.update({f"{population}.{weighting}.{k}": v for k, v in values.items()})
        for k in (
            "nll_in_catalog",
            "nll_user_macro_in_catalog",
            "acquisitions_mean",
            "out_of_catalog",
            "in_catalog_requests",
        ):
            row[k] = summary.get(k)
        timing = path.parent / "latency.json"
        if timing.exists():
            repetitions = read_json(timing)["repetitions"]
            for key in ("p50_ms", "p95_ms"):
                row[key] = float(np.mean([x[key] for x in repetitions]))
        records.append(row)
        parts = path.relative_to(root).parts
        # Keep every scenario directory distinct; collapse exactly the seed directory.
        key = "/".join(p for p in parts[:-1] if not (p.startswith("seed") and p[4:].isdigit()))
        groups[key].append(row)
    columns = [
        "path",
        "method",
        "seed",
        "population",
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@10",
        "mrr",
        "cost",
        "requests",
    ]
    columns += sorted({k for r in records for k in r} - set(columns))
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)
    aggregate = {}
    for k, rows in groups.items():
        seeds = [r["seed"] for r in rows if r["seed"] is not None]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate seed result in group {k}")
        aggregate[k] = dict(
            n=len(rows),
            seeds=seeds,
            population=rows[0]["population"],
            metrics={
                metric: dict(
                    mean=float(np.mean(values)),
                    std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    n=len(values),
                )
                for metric in columns
                if metric not in ("path", "method", "seed", "population")
                if (values := [r[metric] for r in rows if r.get(metric) is not None])
            },
        )
    write_json(out / "aggregate.json", aggregate)
    lines = [
        "# Completed experiment results",
        "",
        "Single-run standard deviation is omitted. Missing experiments have no fabricated entries.",
        "",
        "| Scenario | Population | Runs | NDCG@10 mean ± sample std | Mean cost |",
        "|---|---|---:|---:|---:|",
    ]
    for key, x in aggregate.items():
        nd = x["metrics"]["ndcg@10"]
        sd = "—" if nd["std"] is None else f"{nd['std']:.5f}"
        lines.append(
            f"| {key} | {x['population']} | {x['n']} | {nd['mean']:.5f} ± {sd} | {x['metrics']['cost']['mean']:.4f} |"
        )
    (out / "results.md").write_text("\n".join(lines) + "\n")
