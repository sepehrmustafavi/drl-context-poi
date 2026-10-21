"""Executable experiment DAG and independent-GPU workers (no DDP required)."""

from pathlib import Path
import json
import os
import socket
import subprocess
import sys
import time
import numpy as np
import yaml
from .util import read_json, write_json, digest, file_hash

SEEDS = (11, 29, 47)


def plan(data_root, out, suite="full", settings_dir=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(Path("configs/base.yaml").read_text())
    jobs = []

    def add(id, argv, deps=(), kind="gpu"):
        jobs.append(dict(id=id, argv=argv, dependencies=list(deps), kind=kind))
        return id

    def conf(name, cfg):
        path = out / "configs" / f"{name}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return str(path)

    def policies(prefix, cpath, cost, caches, fitid, cacheids, methods):
        for method in methods:
            pd = f"runs/{prefix}/policies/{method}"
            tid = add(
                f"{prefix}/{method}/train",
                [
                    "train-policy",
                    "--resume",
                    "--config",
                    cpath,
                    "--cost-profile",
                    cost,
                    "--train-cache",
                    caches["policy"],
                    "--val-cache",
                    caches["val"],
                    "--method",
                    method,
                    "--out",
                    pd,
                ],
                [cacheids["policy"], cacheids["val"]],
            )
            add(
                f"{prefix}/{method}/test",
                [
                    "evaluate",
                    "--config",
                    cpath,
                    "--cost-profile",
                    cost,
                    "--cache",
                    caches["test"],
                    "--policy",
                    pd + "/policy.pt",
                    "--out",
                    pd + "/test",
                ],
                [tid, cacheids["test"]],
            )

    for dataset in ("nyc", "tky", "gowalla_ca"):
        csv = str(Path(data_root) / (dataset + ".csv"))
        ds = dict(base)
        if settings_dir:
            ds = yaml.safe_load((Path(settings_dir) / (dataset + ".yaml")).read_text())
        if dataset == "gowalla_ca":
            ds.update(
                sources=["remote", "periodic", "transition"],
                fixed_order=[0, 1, 2],
                max_acquisitions=min(ds["max_acquisitions"], 3),
            )
        for backbone in (
            ("transformer", "gru") if suite == "full" and dataset != "gowalla_ca" else ("transformer",)
        ):
            seeds = SEEDS if backbone == "transformer" else (SEEDS[0],)
            for seed in seeds:
                cfg = dict(ds, seed=seed, backbone=backbone, ensemble=3)
                prefix = f"{dataset}/{backbone}/seed{seed}"
                cpath = conf(prefix.replace("/", "_"), cfg)
                pred = f"runs/{prefix}/predictor"
                fid = add(
                    prefix + "/fit", ["fit", "--csv", csv, "--config", cpath, "--out", pred, "--resume"]
                )
                cost = f"runs/{dataset}/transformer/reference_costs.json"
                costid = f"{dataset}/transformer/costs"
                if seed == seeds[0] and backbone == "transformer":
                    if cfg.get("costs") is not None:
                        if "cost_profile_data_identity" not in cfg:
                            raise ValueError("Frozen costs require their data identity from controller-hpo")
                        frozen = (
                            read_json(cfg["frozen_cost_profile"])
                            if cfg.get("frozen_cost_profile")
                            else dict(
                                costs=cfg["costs"],
                                sources=cfg["sources"],
                                data_identity=cfg["cost_profile_data_identity"],
                            )
                        )
                        if (
                            frozen["costs"] != cfg["costs"]
                            or frozen["data_identity"] != cfg["cost_profile_data_identity"]
                        ):
                            raise ValueError("Frozen cost profile changed after selection")
                        write_json(cost, dict(frozen, frozen_before_test=True))
                        add(
                            costid,
                            ["audit", "--csv", csv, "--config", cpath, "--out", cost + ".audit.json"],
                            [fid],
                            kind="cpu",
                        )
                    else:
                        add(
                            costid,
                            [
                                "profile-cost",
                                "--csv",
                                csv,
                                "--config",
                                cpath,
                                "--predictor",
                                pred,
                                "--out",
                                cost,
                            ],
                            [fid],
                            kind="reference",
                        )
                caches = {}
                cacheids = {}
                for split in ("policy", "val", "test"):
                    caches[split] = f"runs/{prefix}/cache/{split}"
                    cacheids[split] = add(
                        prefix + "/cache/" + split,
                        [
                            "cache",
                            "--csv",
                            csv,
                            "--config",
                            cpath,
                            "--predictor",
                            pred,
                            "--split",
                            split,
                            "--out",
                            caches[split],
                        ],
                        [fid],
                    )
                # Policy training depends on both caches AND the fixed cost reference.
                policydeps = dict(cacheids)
                barrier = add(
                    prefix + "/ready",
                    ["audit", "--csv", csv, "--config", cpath, "--out", f"runs/{prefix}/audit.json"],
                    [costid, *cacheids.values()],
                    kind="cpu",
                )
                for split in policydeps:
                    policydeps[split] = barrier
                methods = (
                    ["ddqn", "dqn", "myopic", "direct", "bandit"] if backbone == "transformer" else ["ddqn"]
                )
                policies(prefix, cpath, cost, caches, fid, policydeps, methods)
                for method in ("base", "all", "fixed", "entropy", "random", "oracle"):
                    add(
                        prefix + "/" + method + "/test",
                        [
                            "evaluate",
                            "--config",
                            cpath,
                            "--cost-profile",
                            cost,
                            "--cache",
                            caches["test"],
                            "--val-cache",
                            caches["val"],
                            "--method",
                            method,
                            "--out",
                            f"runs/{prefix}/references/{method}/test",
                        ],
                        [barrier],
                        kind="cpu",
                    )
                if backbone == "transformer" and seed == SEEDS[0]:
                    for personal in (False, True):
                        add(
                            prefix + ("/personal" if personal else "/popular"),
                            [
                                "popularity",
                                "--csv",
                                csv,
                                "--config",
                                cpath,
                                "--out",
                                f"runs/{prefix}/references/"
                                + ("personal" if personal else "popularity")
                                + "/test",
                            ]
                            + (["--personal"] if personal else []),
                            kind="cpu",
                        )
                    add(
                        prefix + "/external-export",
                        [
                            "export-external",
                            "--csv",
                            csv,
                            "--config",
                            cpath,
                            "--out",
                            f"runs/{dataset}/external",
                        ],
                        kind="cpu",
                    )
                    if suite == "full":
                        policies(
                            prefix,
                            cpath,
                            cost,
                            caches,
                            fid,
                            policydeps,
                            ["no_stop", "no_uncertainty", "no_representation"],
                        )
                        variants = []
                        if dataset != "gowalla_ca":
                            variants += [("leave_" + s, dict(disabled_sources=[s])) for s in cfg["sources"]]
                            variants += [
                                ("budget_" + str(v), dict(budget_fraction=v)) for v in (0, 0.25, 0.5, 0.75, 1)
                            ]
                            variants += [
                                ("lambda_" + str(v), dict(lambda_cost=v)) for v in (0, 0.03, 0.1, 0.3, 1)
                            ]
                            variants += [
                                ("cap_" + str(v), dict(max_acquisitions=v))
                                for v in range(1, len(cfg["sources"]) + 1)
                            ]
                        for name, change in variants:
                            vc = conf(prefix.replace("/", "_") + "_" + name, dict(cfg, **change))
                            policies(prefix + "/" + name, vc, cost, caches, fid, policydeps, ["ddqn"])
                # M=1 reuse member 0 parameters: original member0 is trained independently.
                # A separate refit below ensures directory/config/checkpoint provenance remains coherent.
                if (
                    suite == "full"
                    and backbone == "transformer"
                    and dataset != "gowalla_ca"
                    and seed == SEEDS[0]
                ):
                    single = dict(cfg, ensemble=1)
                    sp = f"{dataset}/single/seed{seed}"
                    sc = conf(sp.replace("/", "_"), single)
                    spr = f"runs/{sp}/predictor"
                    sf = add(sp + "/fit", ["fit", "--csv", csv, "--config", sc, "--out", spr, "--resume"])
                    sco, scoid = cost, costid
                    sca = {}
                    sids = {}
                    for split in ("policy", "val", "test"):
                        sca[split] = f"runs/{sp}/cache/{split}"
                        sids[split] = add(
                            sp + "/cache/" + split,
                            [
                                "cache",
                                "--csv",
                                csv,
                                "--config",
                                sc,
                                "--predictor",
                                spr,
                                "--split",
                                split,
                                "--out",
                                sca[split],
                            ],
                            [sf, scoid],
                        )
                    policies(sp, sc, sco, sca, sf, sids, ["ddqn"])
                if backbone == "transformer":
                    for name in ("sasrec", "getnext", "dchl", "ipcm"):
                        add(
                            f"{dataset}/{name}/seed{seed}/baseline",
                            [
                                "baseline",
                                "--csv",
                                csv,
                                "--config",
                                cpath,
                                "--model",
                                name,
                                "--out",
                                f"runs/{dataset}/{name}/seed{seed}",
                                "--resume",
                            ],
                        )
    # Keep timing in a separate phase on one idle reference device.
    fits = {j["argv"][j["argv"].index("--out") + 1]: j for j in jobs if j["argv"][0] == "fit"}
    caches = {j["argv"][j["argv"].index("--out") + 1]: j for j in jobs if j["argv"][0] == "cache"}
    for job in list(jobs):
        argv = job["argv"]
        if argv[0] == "profile-cost":
            job["kind"] = "reference"
        if argv[0] != "evaluate" or "/seed11/" not in job["id"]:
            continue

        def arg(k):
            return argv[argv.index(k) + 1] if k in argv else None

        method = arg("--method")
        if method == "oracle":
            continue  # label-dependent oracle is not deployable
        cachejob = caches[arg("--cache")]
        cav = cachejob["argv"]
        pred = cav[cav.index("--predictor") + 1]
        csv = cav[cav.index("--csv") + 1]
        timing = [
            "latency",
            "--csv",
            csv,
            "--predictor",
            pred,
            "--config",
            arg("--config"),
            "--cost-profile",
            arg("--cost-profile"),
            "--out",
            arg("--out") + "/latency.json",
        ]
        if arg("--policy"):
            timing += ["--policy", arg("--policy")]
        else:
            timing += ["--method", method]
        if method == "fixed":
            timing += ["--selection", arg("--out") + "/summary.json"]
        add(job["id"] + "/latency", timing, [job["id"]], kind="timing")
        if "/transformer/seed11/" in job["id"] and ("/ddqn/test" in job["id"] or method in ("base", "all")):
            if job["id"].startswith(("nyc/", "tky/")):
                add(
                    job["id"] + "/subgroups",
                    [
                        "subgroups",
                        "--csv",
                        csv,
                        "--config",
                        arg("--config"),
                        "--results",
                        arg("--out") + "/requests.jsonl",
                        "--out",
                        arg("--out") + "/subgroups.json",
                    ],
                    [job["id"]],
                    kind="cpu",
                )
    for job in list(jobs):
        if job["argv"][0] == "baseline" and "/seed11/" in job["id"]:
            argv = job["argv"]

            def value(k):
                return argv[argv.index(k) + 1]

            add(
                job["id"] + "/latency",
                [
                    "baseline-latency",
                    "--csv",
                    value("--csv"),
                    "--config",
                    value("--config"),
                    "--model",
                    value("--model"),
                    "--checkpoint",
                    value("--out") + "/model.pt",
                    "--out",
                    value("--out") + "/test/latency.json",
                ],
                [job["id"]],
                kind="timing",
            )
    for job in list(jobs):
        av = job["argv"]
        if av[0] != "popularity":
            continue

        def value(k):
            return av[av.index(k) + 1]

        add(
            job["id"] + "/latency",
            [
                "frequency-latency",
                "--csv",
                value("--csv"),
                "--config",
                value("--config"),
                "--out",
                value("--out") + "/latency.json",
            ]
            + (["--personal"] if "--personal" in av else []),
            [job["id"]],
            kind="timing",
        )
    for pred, job in fits.items():
        if "/transformer/seed11/" not in pred:
            continue
        av = job["argv"]
        csv = av[av.index("--csv") + 1]
        cp = av[av.index("--config") + 1]
        dataset = job["id"].split("/")[0]
        add(
            job["id"] + "/complexity",
            [
                "complexity",
                "--csv",
                csv,
                "--config",
                cp,
                "--predictor",
                pred,
                "--cost-profile",
                f"runs/{dataset}/transformer/reference_costs.json",
                "--out",
                pred + "/complexity.json",
            ],
            [job["id"], f"{dataset}/transformer/costs"],
            kind="timing",
        )
    for dataset in ("nyc", "tky", "gowalla_ca"):
        comparisons = []
        for other in ("dqn", "myopic", "direct", "bandit", "fixed", "entropy", "random", "base", "all"):
            kind = "policies" if other in ("dqn", "myopic", "direct", "bandit") else "references"
            comparisons.append(
                dict(
                    name="ddqn vs " + other,
                    a=[
                        f"runs/{dataset}/transformer/seed{s}/policies/ddqn/test/requests.jsonl" for s in SEEDS
                    ],
                    b=[
                        f"runs/{dataset}/transformer/seed{s}/{kind}/{other}/test/requests.jsonl"
                        for s in SEEDS
                    ],
                )
            )
        for other in ("sasrec", "getnext", "dchl", "ipcm"):
            comparisons.append(
                dict(
                    name="ddqn vs " + other,
                    a=[
                        f"runs/{dataset}/transformer/seed{s}/policies/ddqn/test/requests.jsonl" for s in SEEDS
                    ],
                    b=[f"runs/{dataset}/{other}/seed{s}/test/requests.jsonl" for s in SEEDS],
                )
            )
        for other in ("popularity", "personal"):
            comparisons.append(
                dict(
                    name="ddqn vs " + other,
                    a=[
                        f"runs/{dataset}/transformer/seed{s}/policies/ddqn/test/requests.jsonl" for s in SEEDS
                    ],
                    b=[f"runs/{dataset}/transformer/seed11/references/{other}/test/requests.jsonl"],
                )
            )
        write_json(
            out / f"family_{dataset}.json",
            dict(name=dataset + " primary NDCG@10 comparisons", predeclared=True, comparisons=comparisons),
        )
    ids = {j["id"] for j in jobs}
    if len(ids) != len(jobs) or any(d not in ids for j in jobs for d in j["dependencies"]):
        raise AssertionError("Invalid DAG IDs/dependencies")
    write_json(
        out / "plan.json",
        dict(
            source_hashes={str(p): file_hash(p) for p in sorted(Path("src/drlpoi").glob("*.py"))},
            config_hashes={str(p): file_hash(p) for p in sorted((out / "configs").glob("*.yaml"))},
            data_hashes={
                str(Path(data_root) / (d + ".csv")): file_hash(Path(data_root) / (d + ".csv"))
                if (Path(data_root) / (d + ".csv")).exists()
                else None
                for d in ("nyc", "tky", "gowalla_ca")
            },
            version=1,
            suite=suite,
            jobs=jobs,
            publication_gaps=[
                "Official-class baseline adapters require the baselines extra and pinned upstream sources",
                "Run timing worker separately with --timing on idle reference GPU 0 after all training",
                "HPO selection must be frozen before generating/running the final plan",
            ],
            seeds=list(SEEDS),
            settings_dir=settings_dir,
            note="All paths relative to repository root; execute workers from that root",
        ),
    )
    print(f"Created {len(jobs)} executable jobs. No experiments have been run.")


def worker(plan_path, gpu="0", cpu=False, once=False, timing=False):
    plan = read_json(plan_path)
    for p, expected in {**plan.get("config_hashes", {}), **plan.get("source_hashes", {})}.items():
        if file_hash(p) != expected:
            raise ValueError("Configuration changed after plan generation: " + p)
    for p, expected in plan.get("data_hashes", {}).items():
        if expected is None or file_hash(p) != expected:
            raise ValueError("Data missing or changed; regenerate the plan: " + p)
    root = Path(plan_path).parent / "state"
    root.mkdir(exist_ok=True)
    signature = digest(plan)
    marker = root / "plan_signature.json"
    if marker.exists() and read_json(marker)["sha256"] != signature:
        raise ValueError("Plan changed; use a fresh state directory")
    write_json(marker, dict(sha256=signature))
    owner = f"{socket.gethostname()}:{os.getpid()}"

    def state(job, suffix):
        return root / (digest(job["id"])[:20] + suffix)

    if timing:
        if cpu or str(gpu) != "0":
            raise ValueError("Timing uses only reference GPU 0, one worker")
        if any(not state(j, ".done.json").exists() for j in plan["jobs"] if j["kind"] != "timing"):
            raise ValueError("Complete all non-timing jobs before starting timing")
    eligible = [
        j for j in plan["jobs"] if (j["kind"] == "timing") == timing and (not cpu or j["kind"] == "cpu")
    ]
    while True:
        ran = False
        for job in eligible:
            done, lock, fail = [state(job, x) for x in (".done.json", ".lock", ".failed.json")]
            if done.exists() or fail.exists():
                continue
            if job["kind"] == "reference" and str(gpu) != "0":
                continue
            if not all((root / f"{digest(d)[:20]}.done.json").exists() for d in job["dependencies"]):
                continue
            # Serialize claims so a profiling job never races with a training claim.
            mutex = root / "scheduler.mutex"
            try:
                fd = os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            claimed = False
            try:
                active = list(root.glob("*.lock"))
                exclusive = any(read_json(p).get("kind") in ("reference", "timing") for p in active)
                if exclusive or (job["kind"] in ("reference", "timing") and active):
                    continue
                if done.exists() or lock.exists():
                    continue
                try:
                    lf = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                with os.fdopen(lf, "w") as f:
                    json.dump(dict(job=job["id"], kind=job["kind"], owner=owner, start=time.time()), f)
                claimed = True
            finally:
                os.close(fd)
                mutex.unlink(missing_ok=True)
            if not claimed:
                continue
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
            log = state(job, ".log")
            start = time.time()
            try:
                print(f"Running {job['id']} on {'CPU' if cpu else 'GPU ' + str(gpu)}", flush=True)
                with open(log, "a") as f:
                    proc = subprocess.run(
                        [sys.executable, "-m", "drlpoi.cli", *job["argv"]],
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                write_json(
                    done if proc.returncode == 0 else fail,
                    dict(
                        job=job["id"],
                        seconds=time.time() - start,
                        returncode=proc.returncode,
                        owner=owner,
                        argv=job["argv"],
                    ),
                )
            finally:
                lock.unlink(missing_ok=True)
            ran = True
            if proc.returncode:
                raise RuntimeError(f"Job failed: {job['id']}; inspect {log}; fix and remove its .failed.json")
            if once:
                return
        pending = [j for j in eligible if not state(j, ".done.json").exists()]
        if not pending:
            print("Selected phase complete.")
            return
        if not ran:
            if list(root.glob("*.lock")) or (root / "scheduler.mutex").exists():
                time.sleep(2)
                continue
            raise RuntimeError(
                "Pending jobs are blocked by failed jobs or require reference GPU 0; inspect state"
            )


def hpo(csv, cfg, out):
    """Bounded, single-seed predictor HPO. No test evaluation or reduced data pilot."""
    from .predictor import fit

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = cfg.get(
        "predictor_search",
        [
            dict(dim=d, layers=layers, learning_rate=lr)
            for d, layers in ((64, 1), (64, 2), (128, 2))
            for lr in (0.0003, 0.001)
        ],
    )
    if not candidates or cfg["seed"] != 11:
        raise ValueError("Predeclare nonempty predictor search with fixed seed 11")
    write_json(out / "search.json", dict(candidates=candidates, seed=11, split="val"))
    rows = []
    for i, change in enumerate(candidates):
        cc = dict(cfg, **change)
        path = out / f"candidate_{i}"
        fit(csv, cc, path, resume=True)
        from .predictor import load_ensemble
        from .model import pack, ensemble_predict
        from .metrics import rank_of, per_request

        st, models, device = load_ensemble(csv, path, cc["device"])
        validation = st.split("val")
        scores = []
        for start in range(0, len(validation), cc["batch_size"]):
            rs = validation[start : start + cc["batch_size"]]
            conditions = []
            for full in (False, True):
                masks = np.full((len(rs), len(st.sources)), full)
                probabilities = ensemble_predict(models, pack(st, rs, masks, device))[0].cpu().numpy()
                conditions.append(
                    [per_request(rank_of(p, st.target(r)))["ndcg@10"] for r, p in zip(rs, probabilities)]
                )
            scores.extend(np.mean(conditions, axis=0))
        rows.append(dict(candidate=i, parameters=change, validation_ndcg=float(np.mean(scores))))
    if not rows:
        raise ValueError("No HPO candidates")
    selected = max(rows, key=lambda x: x["validation_ndcg"])
    final = dict(cfg, **selected["parameters"])
    (out / "selected.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    write_json(
        out / "selection.json",
        dict(
            candidates=rows,
            selected=selected,
            seed=cfg["seed"],
            criterion="ensemble validation NDCG over empty/all contexts, all requests; same full chronological dataset",
            note="Predictor-only HPO. Policy budget/lambda/threshold must also be fixed using validation.",
        ),
    )


def controller_hpo(train_cache, val_cache, cfg, out):
    """Predeclared bounded controller grid; never opens a test cache."""
    from .cache import Cache
    from .policy import train, evaluate
    from .util import load_torch

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg["seed"] != 11:
        raise ValueError("Controller search uses fixed seed 11")
    tr, va = Cache(train_cache, cfg), Cache(val_cache, cfg)
    tr.compatible(va)
    if tr.meta["identity"]["split"] != "policy" or va.meta["identity"]["split"] != "val":
        raise ValueError("HPO requires policy-fit and validation caches")
    candidates = cfg.get(
        "controller_search",
        [
            dict(q_hidden=64, q_lr=0.001, replay_capacity=25000, target_interval=100, lambda_cost=0.03),
            dict(q_hidden=128, q_lr=0.0003, replay_capacity=50000, target_interval=250, lambda_cost=0.1),
            dict(q_hidden=256, q_lr=0.0001, replay_capacity=100000, target_interval=500, lambda_cost=0.3),
            dict(q_hidden=128, q_lr=0.001, replay_capacity=100000, target_interval=100, lambda_cost=1.0),
            dict(q_hidden=64, q_lr=0.0003, replay_capacity=50000, target_interval=500, lambda_cost=0.1),
            dict(q_hidden=256, q_lr=0.0003, replay_capacity=25000, target_interval=250, lambda_cost=0.03),
        ],
    )
    if not candidates:
        raise ValueError("Empty controller search")
    write_json(
        out / "search.json",
        dict(
            candidates=candidates,
            seed=11,
            predictor_hashes=tr.meta["identity"]["model_hashes"],
            criterion="validation NDCG@10, cost tie break",
        ),
    )
    rows = []
    for i, change in enumerate(candidates):
        cc = dict(cfg, **change)
        path = out / f"candidate_{i}"
        train(train_cache, val_cache, cc, "ddqn", path, resume=True)
        ck = load_torch(path / "policy.pt")
        rows.append(
            dict(
                candidate=i,
                parameters=change,
                ndcg=ck["val"]["all"]["request_macro"]["ndcg@10"],
                cost=ck["val"]["cost_mean"],
            )
        )
    selected = max(rows, key=lambda r: (r["ndcg"], -r["cost"]))
    final = dict(cfg, **selected["parameters"])
    final["cost_profile_data_identity"] = tr.meta["identity"]["data"]
    thresholds = cfg.get("entropy_search", [0.0, 0.25, 0.5, 0.7, 0.9, 1.0])
    entropy = []
    for value in thresholds:
        _, result = evaluate(Cache(val_cache, dict(final, entropy_threshold=value)), "entropy")
        entropy.append(
            dict(threshold=value, ndcg=result["all"]["request_macro"]["ndcg@10"], cost=result["cost_mean"])
        )
    if not entropy:
        raise ValueError("Empty entropy threshold search")
    final["entropy_threshold"] = max(entropy, key=lambda r: (r["ndcg"], -r["cost"]))["threshold"]
    (out / "selected.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    write_json(
        out / "selection.json",
        dict(
            candidates=rows,
            selected=selected,
            entropy=entropy,
            seed=11,
            budget_fraction=cfg["budget_fraction"],
            max_acquisitions=cfg["max_acquisitions"],
            train_cache=tr.meta["signature"],
            val_cache=va.meta["signature"],
        ),
    )


def calibrate(csv, cfg, out, export_path):
    """Full-data, validation-only selection followed by a frozen configuration.

    No test cache, test metrics or pilot data are read by this entry point.
    """
    from .cache import build
    from .runtime import profile_cost
    from .util import read_json

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if cfg["seed"] != 11:
        raise ValueError("Selection is predeclared at seed 11")
    hpo(csv, cfg, out / "predictor_search")
    selection = read_json(out / "predictor_search/selection.json")
    chosen = selection["selected"]["candidate"]
    predictor = out / "predictor_search" / f"candidate_{chosen}"
    chosen_cfg = yaml.safe_load((out / "predictor_search/selected.yaml").read_text())
    costpath = out / "reference_costs.json"
    if not costpath.exists():
        profile_cost(csv, predictor, costpath, device=cfg["device"])
    costs = read_json(costpath)
    from .data import Store
    from .util import file_hash

    if costs["data_identity"] != Store(csv, chosen_cfg).identity:
        raise ValueError("Calibration cost profile data mismatch")
    if costs["predictor_hashes"] != [
        file_hash(predictor / f"member_{i}.pt") for i in range(chosen_cfg["ensemble"])
    ]:
        raise ValueError("Calibration cost profile predictor mismatch")
    chosen_cfg.update(
        costs=costs["costs"],
        cost_profile_data_identity=costs["data_identity"],
        frozen_cost_profile=str(costpath),
    )
    for split in ("policy", "val"):
        build(csv, predictor, out / "cache" / split, split, chosen_cfg["device"])
    controller_hpo(out / "cache/policy", out / "cache/val", chosen_cfg, out / "controller_search")
    target = Path(export_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text((out / "controller_search/selected.yaml").read_text())
    write_json(
        str(target) + ".selection.json",
        dict(
            predictor_selection=str(out / "predictor_search/selection.json"),
            controller_selection=str(out / "controller_search/selection.json"),
            cost_profile=str(costpath),
            selected_config_sha256=file_hash(target),
            test_evaluated=False,
        ),
    )
