import json
import numpy as np
import pytest
from drlpoi.predictor import fit, load_ensemble
from drlpoi.cache import build, Cache
from drlpoi.policy import train, load_policy, evaluate, make_transition, LEARNED
from drlpoi.runtime import Inference, latency, profile_cost
from drlpoi.external import export, import_scores
from drlpoi.util import write_json


def test_complete_pipeline(csv_path, cfg, tmp_path, monkeypatch):
    pred = tmp_path / "predictor"
    fit(csv_path, cfg, pred)
    caches = {s: tmp_path / s for s in ("policy", "val", "test")}
    for s, path in caches.items():
        build(csv_path, pred, path, s, "cpu")
    tr = Cache(caches["policy"], cfg)
    te = Cache(caches["test"], cfg)
    for i in range(len(tr.requests)):
        mask = 0
        reward = 0
        while True:
            valid = tr.valid(i, mask)
            choices = np.flatnonzero(valid[:-1])
            a = int(choices[0]) if len(choices) else tr.j
            transition, new = make_transition(tr, i, mask, a)
            reward += transition[2]
            mask = new
            if transition[4]:
                break
        assert np.isclose(reward, tr.objective(i, mask))
    for method in LEARNED:
        out = tmp_path / method
        train(caches["policy"], caches["val"], cfg, method, out)
        net, loaded = load_policy(out / "policy.pt", te)
        rows, summary = evaluate(te, loaded, net)
        assert summary["requests"] == len(te.requests)
        if method == "ddqn":
            before = (out / "policy.pt").read_bytes()
            train(caches["policy"], caches["val"], cfg, method, out, resume=True)
            assert (out / "policy.pt").read_bytes() == before
        assert all(row["cost"] <= te.budget + 1e-6 for row in rows)
        # Raw lazy inference must reproduce cached policy decisions and exact rankings.
        engine = Inference(csv_path, pred, cfg, out / "policy.pt")
        for i, r in enumerate(engine.store.split("test")[:3]):
            p, m, trace = engine.run(r)
            assert m == rows[i]["mask"] and trace == rows[i]["trace"]
            if engine.store.target(r) >= 0:
                from drlpoi.metrics import rank_of

                assert rank_of(p, engine.store.target(r)) == int(te.rank[i, m])

    # Simulate an interruption after a committed epoch, then verify identical continuation.
    import drlpoi.policy as policy_module
    from drlpoi.util import load_torch

    two = dict(cfg, policy_epochs=2)
    continuous = tmp_path / "continuous"
    interrupted = tmp_path / "interrupted"
    train(caches["policy"], caches["val"], two, "ddqn", continuous)
    original_save = policy_module.save_torch

    def interrupt_after_commit(path, obj):
        original_save(path, obj)
        if path.name == "last.pt" and obj["epoch"] == 0:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(policy_module, "save_torch", interrupt_after_commit)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train(caches["policy"], caches["val"], two, "ddqn", interrupted)
    train(caches["policy"], caches["val"], two, "ddqn", interrupted, resume=True)
    a = load_torch(continuous / "last.pt")["model"]
    b = load_torch(interrupted / "last.pt")["model"]
    import torch

    assert all(torch.equal(a[k], b[k]) for k in a)

    zero = dict(cfg, budget_fraction=0)
    zc = Cache(caches["test"], zero)
    assert list(zc.feasible(0)) == [0]
    assert zc.valid(0, 0).tolist() == [False] * zc.j + [True]
    assert np.isfinite(zc.state(0, 0)).all()
    out = tmp_path / "latency.json"
    latency(csv_path, pred, cfg, out, method="base", count=3, warmup=1, repeats=1)
    profile_cost(csv_path, pred, tmp_path / "costs.json", count=5, device="cpu")
    manifest = export(csv_path, cfg, tmp_path / "exchange")
    scores = tmp_path / "scores.jsonl"
    store, _, _ = load_ensemble(csv_path, pred, "cpu")
    with open(scores, "w") as f:
        for r in store.split("test"):
            f.write(json.dumps(dict(request_id=r.id, scores=[0.0] * len(store.catalog))) + "\n")
    provenance = dict(
        data_identity_sha256=manifest["data_identity_sha256"],
        catalog_sha256=manifest["catalog_sha256"],
        request_ids_sha256=manifest["request_ids_sha256"]["test"],
        source_commit="test-fixture",
        model_name="fixture-only",
        seed=11,
        training_config={},
        train_split="fit",
        target_time_used=False,
        temporal_leakage_audited=True,
    )
    write_json(tmp_path / "provenance.json", provenance)
    rows, summary = import_scores(csv_path, cfg, scores, tmp_path / "provenance.json")
    assert summary["out_of_catalog"] == 1
    provenance["target_time_used"] = True
    write_json(tmp_path / "provenance.json", provenance)
    with pytest.raises(ValueError, match="leakage"):
        import_scores(csv_path, cfg, scores, tmp_path / "provenance.json")
