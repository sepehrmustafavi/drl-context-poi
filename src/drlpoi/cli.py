from __future__ import annotations
import argparse
import json
from pathlib import Path
from .util import config, read_json, write_json


def add_config(p):
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    p.add_argument("--cost-profile")


def configured(args):
    cfg = config(args.config, args.set)
    if args.cost_profile:
        profile = read_json(args.cost_profile)
        if profile["sources"] != cfg["sources"]:
            raise ValueError("Source order differs from cost profile")
        cfg["costs"] = profile["costs"]
        cfg["cost_profile_data_identity"] = profile["data_identity"]
        if hasattr(args, "csv") and args.csv:
            from .data import Store

            if profile["data_identity"] != Store(args.csv, cfg).identity:
                raise ValueError("Cost profile belongs to another dataset/configuration")
    return cfg


def dump_results(out, rows, summary):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "requests.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Budgeted context acquisition for next-POI recommendation")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("convert")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--format", choices=["canonical", "foursquare", "gowalla", "getnext-gowalla"], required=True
    )
    p.add_argument("--bbox", nargs=4, type=float, metavar=("SOUTH", "NORTH", "WEST", "EAST"))
    for cmd in (
        "audit",
        "fit",
        "cache",
        "profile-cost",
        "train-policy",
        "evaluate",
        "infer",
        "latency",
        "popularity",
        "export-external",
        "import-external",
    ):
        p = subs.add_parser(cmd)
        add_config(p)
        if cmd not in ("train-policy", "evaluate"):
            p.add_argument("--csv", required=True)
        if cmd in (
            "fit",
            "cache",
            "profile-cost",
            "train-policy",
            "evaluate",
            "infer",
            "latency",
            "popularity",
            "export-external",
            "import-external",
            "audit",
        ):
            p.add_argument("--out", required=True)
        if cmd in ("cache", "profile-cost", "infer", "latency"):
            p.add_argument("--predictor", required=True)
        if cmd == "fit":
            p.add_argument("--resume", action="store_true")
        if cmd == "cache":
            p.add_argument("--split", choices=["policy", "val", "test"], required=True)
        if cmd == "profile-cost":
            p.add_argument("--count", type=int, default=100)
        if cmd == "train-policy":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--train-cache", required=True)
            p.add_argument("--val-cache", required=True)
            from .policy import LEARNED

            p.add_argument("--method", choices=LEARNED, required=True)
        if cmd == "evaluate":
            p.add_argument("--cache", required=True)
            p.add_argument("--val-cache")
            p.add_argument("--policy")
            p.add_argument("--method", default="ddqn")
        if cmd in ("infer", "latency"):
            p.add_argument("--policy")
            p.add_argument("--method")
            p.add_argument("--fixed-mask", type=int, default=0)
        if cmd == "latency":
            p.add_argument("--count", type=int, default=1000)
            p.add_argument("--warmup", type=int, default=50)
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--selection")
        if cmd == "popularity":
            p.add_argument("--personal", action="store_true")
        if cmd == "import-external":
            p.add_argument("--scores", required=True)
            p.add_argument("--manifest", required=True)
    p = subs.add_parser("baseline")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", choices=["sasrec", "getnext", "dchl", "ipcm"], required=True)
    p.add_argument("--upstream-root")
    p.add_argument("--resume", action="store_true")
    p = subs.add_parser("calibrate")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--export", required=True)
    p = subs.add_parser("controller-hpo")
    add_config(p)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("frequency-latency")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--personal", action="store_true")
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p = subs.add_parser("complexity")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--predictor", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p = subs.add_parser("compare-family")
    p.add_argument("--family", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=2000)
    p = subs.add_parser("diagnostics")
    p.add_argument("--root", default="runs")
    p.add_argument("--out", required=True)
    p.add_argument("--tolerance", type=float, default=0.002)
    p = subs.add_parser("baseline-latency")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", choices=["sasrec", "getnext", "dchl", "ipcm"], required=True)
    p.add_argument("--upstream-root")
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p = subs.add_parser("compare")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--samples", type=int, default=2000)
    p = subs.add_parser("plan")
    p.add_argument("--data-root", default="data/canonical")
    p.add_argument("--out", default="experiments")
    p.add_argument("--settings-dir")
    p.add_argument("--suite", choices=["main", "full"], default="full")
    p = subs.add_parser("worker")
    p.add_argument("--plan", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--timing", action="store_true")
    p = subs.add_parser("report")
    p.add_argument("--root", default="runs")
    p.add_argument("--out", default="artifacts/report")
    p = subs.add_parser("hpo")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("figures")
    p.add_argument("--aggregate", required=True)
    p.add_argument("--out", required=True)
    p = subs.add_parser("subgroups")
    add_config(p)
    p.add_argument("--csv", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    cmd = args.command
    if cmd == "figures":
        from .figures import figures

        figures(args.aggregate, args.out)
        return
    if cmd == "convert":
        from .data import convert

        convert(args.raw, args.out, args.format, args.bbox)
        return
    if cmd == "plan":
        from .experiments import plan

        plan(args.data_root, args.out, args.suite, args.settings_dir)
        return
    if cmd == "worker":
        from .experiments import worker

        worker(args.plan, args.gpu, args.cpu, args.once, args.timing)
        return
    if cmd == "report":
        from .report import report

        report(args.root, args.out)
        return
    if cmd == "compare-family":
        from .evaluation import compare_family

        compare_family(args.family, args.out, args.samples)
        return
    if cmd == "diagnostics":
        from .evaluation import diagnostics

        diagnostics(args.root, args.out, args.tolerance)
        return
    if cmd == "compare":
        from .metrics import paired_user_bootstrap

        def read(path):
            with open(path) as f:
                return [json.loads(line) for line in f]

        write_json(args.out, paired_user_bootstrap(read(args.a), read(args.b), samples=args.samples))
        return
    cfg = configured(args)
    if cmd == "frequency-latency":
        from .baselines import frequency_latency

        frequency_latency(args.csv, cfg, args.out, args.personal, args.count, args.warmup, args.repeats)
    elif cmd == "calibrate":
        from .experiments import calibrate

        calibrate(args.csv, cfg, args.out, args.export)
    elif cmd == "complexity":
        from .runtime import complexity

        complexity(args.csv, args.predictor, cfg, args.out, args.count, args.warmup, args.repeats)
    elif cmd == "baseline-latency":
        from .upstream import baseline_latency

        baseline_latency(
            args.csv,
            cfg,
            args.model,
            args.checkpoint,
            args.out,
            args.upstream_root,
            args.count,
            args.warmup,
            args.repeats,
        )
    elif cmd == "baseline":
        from .upstream import baseline

        baseline(args.csv, cfg, args.model, args.out, args.upstream_root, args.resume)
    elif cmd == "controller-hpo":
        from .experiments import controller_hpo

        controller_hpo(args.train_cache, args.val_cache, cfg, args.out)
    elif cmd == "subgroups":
        from .analysis import subgroups

        subgroups(args.csv, cfg, args.results, args.out)
    elif cmd == "audit":
        from .data import Store

        write_json(args.out, Store(args.csv, cfg).audit())
    elif cmd == "fit":
        from .predictor import fit

        fit(args.csv, cfg, args.out, args.resume)
    elif cmd == "cache":
        from .cache import build

        build(args.csv, args.predictor, args.out, args.split, cfg["device"])
    elif cmd == "profile-cost":
        from .runtime import profile_cost

        profile_cost(args.csv, args.predictor, args.out, args.count, cfg["device"])
    elif cmd == "train-policy":
        from .policy import train

        train(args.train_cache, args.val_cache, cfg, args.method, args.out, args.resume)
    elif cmd == "evaluate":
        from .cache import Cache
        from .policy import evaluate, load_policy, best_fixed

        ca = Cache(args.cache, cfg)
        net = None
        method = args.method
        fixed = 0
        if args.policy:
            net, method = load_policy(args.policy, ca)
        if method == "fixed":
            if not args.val_cache:
                raise ValueError("Best fixed subset must be selected on --val-cache")
            va = Cache(args.val_cache, cfg)
            ca.compatible(va)
            if va.meta["identity"]["split"] != "val":
                raise ValueError("Fixed subset selection needs validation requests")
            fixed = best_fixed(va)
        rows, summary = evaluate(ca, method, net, fixed)
        summary.update(
            evaluation_population="in_catalog_only" if method == "oracle" else "all_requests",
            unconstrained=method == "all",
            method=method,
            seed=cfg["seed"],
            cache_signature=ca.meta["signature"],
            config=cfg,
            fixed_mask=fixed,
        )
        dump_results(args.out, rows, summary)
    elif cmd == "infer":
        from .runtime import Inference
        from .metrics import rank_of, per_request, summarize
        from .cache import bits

        engine = Inference(args.csv, args.predictor, cfg, args.policy)
        rows = []
        for r in engine.store.split("test"):
            p, m, tr = engine.run(r, args.method, args.fixed_mask)
            y = engine.store.target(r)
            import numpy as np

            rows.append(
                dict(
                    request_id=r.id,
                    user=r.user,
                    in_catalog=y >= 0,
                    mask=m,
                    trace=tr,
                    cost=float(engine.costs[bits(m, engine.j)].sum()),
                    acquisitions=m.bit_count(),
                    **per_request(rank_of(p, y), -np.log(max(float(p[y]), 1e-30)) if y >= 0 else None),
                )
            )
        summary = summarize(rows)
        summary.update(method=args.method or engine.method, seed=cfg["seed"], config=cfg)
        dump_results(args.out, rows, summary)
    elif cmd == "latency":
        from .runtime import latency

        if args.selection:
            selected = read_json(args.selection)
            if selected.get("method") != "fixed":
                raise ValueError("Expected validation-selected fixed subset summary")
            args.fixed_mask = selected["fixed_mask"]

        latency(
            args.csv,
            args.predictor,
            cfg,
            args.out,
            args.policy,
            args.method,
            args.count,
            args.warmup,
            args.repeats,
            args.fixed_mask,
        )
    elif cmd == "popularity":
        from .baselines import popularity

        rows, summary = popularity(args.csv, cfg, args.personal)
        summary.update(seed=cfg["seed"], method="personal" if args.personal else "popularity")
        dump_results(args.out, rows, summary)
    elif cmd == "export-external":
        from .external import export

        export(args.csv, cfg, args.out)
    elif cmd == "import-external":
        from .external import import_scores

        rows, summary = import_scores(args.csv, cfg, args.scores, args.manifest)
        provenance = read_json(args.manifest)
        summary.update(external=provenance, seed=provenance["seed"], method=provenance["model_name"])
        dump_results(args.out, rows, summary)
    elif cmd == "hpo":
        from .experiments import hpo

        hpo(args.csv, cfg, args.out)


if __name__ == "__main__":
    main()
