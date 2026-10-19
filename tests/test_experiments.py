import yaml
from drlpoi.experiments import plan, worker
from drlpoi.util import read_json, write_json
from drlpoi.analysis import holm
from drlpoi.report import report


def test_full_plan_dag_and_policy_seed_counts(tmp_path):
    plan("data/canonical", tmp_path / "exp", "full")
    p = read_json(tmp_path / "exp/plan.json")
    jobs = p["jobs"]
    ids = {x["id"] for x in jobs}
    assert len(ids) == len(jobs)
    assert all(d in ids for j in jobs for d in j["dependencies"])
    for ds in ("nyc", "tky", "gowalla_ca"):
        for seed in (11, 29, 47):
            for method in ("ddqn", "dqn", "myopic", "direct", "bandit"):
                assert f"{ds}/transformer/seed{seed}/{method}/train" in ids
    # Topological traversal catches cycles and dangling dependencies.
    completed = set()
    while len(completed) < len(jobs):
        ready = [j["id"] for j in jobs if j["id"] not in completed and set(j["dependencies"]) <= completed]
        assert ready
        completed.update(ready)
    assert "nyc/transformer/seed11/leave_remote/ddqn/train" in ids
    assert "gowalla_ca/gru/seed11/fit" not in ids
    assert "nyc/gru/seed11/fit" in ids


def test_cpu_worker_executes_and_skips_completed(csv_path, cfg, tmp_path):
    cp = tmp_path / "config.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "audit.json"
    pp = tmp_path / "plan.json"
    write_json(
        pp,
        dict(
            jobs=[
                dict(
                    id="audit",
                    kind="cpu",
                    dependencies=[],
                    argv=["audit", "--config", str(cp), "--csv", str(csv_path), "--out", str(out)],
                )
            ]
        ),
    )
    worker(pp, cpu=True)
    assert out.exists()
    old = out.stat().st_mtime_ns
    worker(pp, cpu=True)
    assert out.stat().st_mtime_ns == old


def test_holm_and_single_seed_report(tmp_path):
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    summary = dict(
        all=dict(
            request_macro={"recall@5": 0.3, "recall@10": 0.4, "recall@20": 0.5, "ndcg@10": 0.2, "mrr": 0.1}
        ),
        cost_mean=0.7,
        requests=100,
        method="ddqn",
        seed=11,
    )
    write_json(tmp_path / "runs/nyc/transformer/seed11/policies/ddqn/test/summary.json", summary)
    report(tmp_path / "runs", tmp_path / "report")
    agg = read_json(tmp_path / "report/aggregate.json")
    x = next(iter(agg.values()))
    assert x["n"] == 1 and x["metrics"]["ndcg@10"]["std"] is None
