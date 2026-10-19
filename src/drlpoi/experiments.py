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
            ds.update(sources=["remote", "periodic", "transition"], fixed_order=[0, 1, 2], max_acquisitions=3)
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
                cost = f"runs/{dataset}/{backbone}/reference_costs.json"
                costid = f"{dataset}/{backbone}/costs"
                if seed == seeds[0]:
                    add(
                        costid,
                        ["profile-cost", "--csv", csv, "--config", cpath, "--predictor", pred, "--out", cost],
                        [fid],
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
                                ("budget_" + str(v), dict(budget_fraction=v)) for v in (0, 0.25, 0.75, 1)
                            ]
                            variants += [("lambda_" + str(v), dict(lambda_cost=v)) for v in (0, 0.03, 0.3, 1)]
                            variants += [
                                ("cap_" + str(v), dict(max_acquisitions=v))
                                for v in range(1, len(cfg["sources"]))
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
                    sco = f"runs/{sp}/costs.json"
                    scoid = add(
                        sp + "/cost",
                        ["profile-cost", "--csv", csv, "--config", sc, "--predictor", spr, "--out", sco],
                        [sf],
                    )
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
                    # Clean SASRec-style causal ID-only transformer, not an external paper's published numbers.
                    sas = dict(cfg, backbone="sasrec", ensemble=1)
                    sn = f"{dataset}/sasrec/seed{seed}"
                    sc = conf(sn.replace("/", "_"), sas)
                    pr = f"runs/{sn}/predictor"
                    sf = add(sn + "/fit", ["fit", "--csv", csv, "--config", sc, "--out", pr, "--resume"])
                    add(
                        sn + "/test",
                        [
                            "infer",
                            "--csv",
                            csv,
                            "--config",
                            sc,
                            "--cost-profile",
                            cost,
                            "--predictor",
                            pr,
                            "--method",
                            "base",
                            "--out",
                            f"runs/{sn}/test",
                        ],
                        [sf, costid],
                    )
    ids = {j["id"] for j in jobs}
    if len(ids) != len(jobs) or any(d not in ids for j in jobs for d in j["dependencies"]):
        raise AssertionError("Invalid DAG IDs/dependencies")
    write_json(
        out / "plan.json",
        dict(
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
                "GETNext/DCHL/iPCM official adapters require independent integration; see docs/external.md",
                "latency runs must be scheduled separately on an idle reference GPU",
                "HPO selection must be frozen before generating/running the final plan",
            ],
            seeds=list(SEEDS),
            settings_dir=settings_dir,
            note="All paths relative to repository root; execute workers from that root",
        ),
    )
    print(f"Created {len(jobs)} executable jobs. No experiments have been run.")


def worker(plan_path, gpu="0", cpu=False, once=False):
    plan = read_json(plan_path)
    for p, expected in plan.get("config_hashes", {}).items():
        if file_hash(p) != expected:
            raise ValueError("Configuration changed after plan generation: " + p)
    for p, expected in plan.get("data_hashes", {}).items():
        if expected is None:
            raise ValueError("Plan was generated before data were available; regenerate after preparing " + p)
        if file_hash(p) != expected:
            raise ValueError("Data changed after plan generation: " + p)
    root = Path(plan_path).parent / "state"
    root.mkdir(exist_ok=True)
    signature = digest(plan)
    marker = root / "plan_signature.json"
    if marker.exists() and read_json(marker)["sha256"] != signature:
        raise ValueError("Plan changed; use a fresh state directory")
    write_json(marker, dict(sha256=signature))
    owner = f"{socket.gethostname()}:{os.getpid()}"
    while True:
        pending = False
        ran = False
        for job in plan["jobs"]:
            key = digest(job["id"])[:20]
            done = root / f"{key}.done.json"
            lock = root / f"{key}.lock"
            fail = root / f"{key}.failed.json"
            if done.exists():
                continue
            pending = True
            if fail.exists():
                continue
            if cpu and job["kind"] != "cpu":
                continue
            if not all((root / f"{digest(d)[:20]}.done.json").exists() for d in job["dependencies"]):
                continue
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            if done.exists():
                os.close(fd)
                lock.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                json.dump(dict(job=job["id"], owner=owner, start=time.time()), f)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
            log = root / f"{key}.log"
            start = time.time()
            returncode = -1
            try:
                print(f"Running {job['id']} on {'CPU' if cpu else 'GPU ' + str(gpu)}", flush=True)
                with open(log, "a") as f:
                    proc = subprocess.run(
                        [sys.executable, "-m", "drlpoi.cli", *job["argv"]],
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    )
                returncode = proc.returncode
                write_json(
                    done if returncode == 0 else fail,
                    dict(
                        job=job["id"],
                        seconds=time.time() - start,
                        returncode=returncode,
                        owner=owner,
                        argv=job["argv"],
                    ),
                )
            finally:
                lock.unlink(missing_ok=True)
            ran = True
            if returncode:
                raise RuntimeError(
                    f"Job failed: {job['id']}; inspect {log}. Fix then delete its .failed.json to retry."
                )
            if once:
                return
        if not pending:
            print("All jobs complete.")
            return
        if not ran:
            print(
                "No unlocked ready jobs for this worker. Pending dependencies/failures/other workers remain."
            )
            return


def hpo(csv, cfg, out):
    """Bounded, single-seed predictor HPO. No test evaluation or reduced data pilot."""
    from .predictor import fit

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    candidates = [(d, lr) for d in (64, 128) for lr in (0.0003, 0.001, 0.003)][: cfg["hpo_candidates"]]
    rows = []
    for i, (d, lr) in enumerate(candidates):
        cc = dict(cfg, dim=d, learning_rate=lr)
        path = out / f"candidate_{i}"
        fit(csv, cc, path, resume=True)
        from .util import load_torch

        scores = [load_torch(path / f"member_{m}.pt")["val_ndcg"] for m in range(cc["ensemble"])]
        rows.append(dict(candidate=i, dim=d, learning_rate=lr, validation_ndcg=float(np.mean(scores))))
    if not rows:
        raise ValueError("No HPO candidates")
    selected = max(rows, key=lambda x: x["validation_ndcg"])
    final = dict(cfg, dim=selected["dim"], learning_rate=selected["learning_rate"])
    (out / "selected.yaml").write_text(yaml.safe_dump(final, sort_keys=False))
    write_json(
        out / "selection.json",
        dict(
            candidates=rows,
            selected=selected,
            seed=cfg["seed"],
            criterion="mean member validation NDCG over empty/all contexts; same full chronological dataset",
            note="Predictor-only HPO. Policy budget/lambda/threshold must also be fixed using validation.",
        ),
    )
