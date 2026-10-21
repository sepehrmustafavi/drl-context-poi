import json
import numpy as np
import pandas as pd
import pytest
import torch
from drlpoi.data import Store, convert
from drlpoi.model import RequestPrediction, ensemble_predict, pack
from drlpoi.predictor import fit, load_ensemble
from drlpoi.cache import build
from drlpoi.experiments import plan, controller_hpo
from drlpoi.util import read_json, write_json
from drlpoi.evaluation import compare_family, diagnostics
from drlpoi.metrics import per_request, paired_user_bootstrap
from drlpoi.runtime import complexity


@pytest.fixture
def prepared(csv_path, cfg, tmp_path):
    pred = tmp_path / "pred"
    fit(csv_path, cfg, pred)
    for split in ("policy", "val", "test"):
        build(csv_path, pred, tmp_path / split, split, "cpu")
    return pred


def test_cached_encoding_equivalence_and_counts(prepared, csv_path, cfg):
    st, models, dev = load_ensemble(csv_path, prepared, "cpu")
    r = st.split("test")[0]
    calls = {"base": 0, "source": 0}
    from unittest.mock import patch

    base, source = models[0].encode_base, models[0].encode_source

    def counted_base(*a, **k):
        calls["base"] += 1
        return base(*a, **k)

    def counted_source(*a, **k):
        calls["source"] += 1
        return source(*a, **k)

    with torch.no_grad():
        reference = {}
        for mask in (0, 1, 3, 7, 15):
            bits = np.array([[(mask >> j) & 1 for j in range(4)]], bool)
            reference[mask] = ensemble_predict(models, pack(st, [r], bits, dev))
        with (
            patch.object(models[0], "encode_base", counted_base),
            patch.object(models[0], "encode_source", counted_source),
        ):
            request = RequestPrediction(st, models, r, dev)
            for mask in reference:
                actual = request.predict(mask)
                assert all(torch.allclose(a, b, atol=2e-6) for a, b in zip(actual, reference[mask]))
        assert calls == {"base": 1, "source": 4}


def test_time_units_indexed_descriptors_and_missing_categories(csv_path, cfg, tmp_path):
    st = Store(csv_path, cfg)
    assert st.t[0] == pd.Timestamp("2020-01-01", tz="UTC").timestamp()
    for r in st.requests:
        assert st.t[r.prefix[-1]] < st.t[r.target]
        av, desc = st.descriptors(r)
        assert np.isclose(desc[1], np.log1p(len(st._history(r, "periodic"))))
    df = pd.read_csv(csv_path, dtype=str)
    df.loc[df.index % 2 == 0, "category"] = "__missing__"
    path = tmp_path / "mixed.csv"
    df.to_csv(path, index=False)
    fit(path, cfg, tmp_path / "missing-predictor")  # category IDs must be contiguous after excluding missing


def test_hpo_freezes_threshold_and_controller(prepared, cfg, tmp_path):
    cfg = dict(
        cfg,
        controller_search=[
            dict(q_hidden=8, q_lr=0.001, replay_capacity=32, target_interval=2, lambda_cost=0.03)
        ],
        entropy_search=[0.0, 0.7, 1.0],
    )
    out = tmp_path / "hpo"
    controller_hpo(tmp_path / "policy", tmp_path / "val", cfg, out)
    result = read_json(out / "selection.json")
    assert len(result["candidates"]) == 1 and len(result["entropy"]) == 3
    with pytest.raises(ValueError, match="policy-fit"):
        controller_hpo(tmp_path / "test", tmp_path / "val", cfg, tmp_path / "bad")


def test_complete_matrix_and_separate_timing(tmp_path, cfg):
    plan("data/canonical", tmp_path / "plan", "full")
    spec = read_json(tmp_path / "plan/plan.json")
    jobs = {j["id"]: j for j in spec["jobs"]}
    for ds in ("nyc", "tky", "gowalla_ca"):
        for seed in (11, 29, 47):
            for name in ("sasrec", "getnext", "dchl", "ipcm"):
                assert f"{ds}/{name}/seed{seed}/baseline" in jobs
            for name in ("base", "all"):
                assert f"{ds}/transformer/seed{seed}/{name}/test" in jobs
        assert jobs[f"{ds}/transformer/costs"]["kind"] == "reference"
    for ds in ("nyc", "tky"):
        for prefix, values in [
            ("budget", (0, 0.25, 0.5, 0.75, 1)),
            ("lambda", (0, 0.03, 0.1, 0.3, 1)),
            ("cap", range(1, 5)),
        ]:
            for v in values:
                key = f"{ds}/transformer/seed11/{prefix}_{v}/ddqn/test"
                assert key in jobs
                assert jobs[key + "/latency"]["kind"] == "timing"
    assert all(j["kind"] != "timing" for j in spec["jobs"] if j["argv"][0] == "train-policy")
    assert len(read_json(tmp_path / "plan/family_nyc.json")["comparisons"]) == 15
    import yaml

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    for ds in ("nyc", "tky", "gowalla_ca"):
        (frozen / (ds + ".yaml")).write_text(yaml.safe_dump(dict(cfg, costs=None, max_acquisitions=1)))
    plan("data/canonical", tmp_path / "frozen-plan", "full", str(frozen))
    selected = yaml.safe_load(
        (tmp_path / "frozen-plan/configs/gowalla_ca_transformer_seed11.yaml").read_text()
    )
    assert selected["max_acquisitions"] == 1


def test_families_and_descriptive_diagnostics(tmp_path):
    row = dict(
        request_id="r1",
        user="u1",
        in_catalog=True,
        cost=1.0,
        acquisitions=1,
        trace=[0, 4],
        **per_request(1, 1.0),
    )
    paths = []
    for seed in (11, 29, 47):
        path = tmp_path / f"seed{seed}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        paths.append(str(path))
    family = tmp_path / "family.json"
    write_json(
        family,
        dict(
            name="predeclared test fixture",
            predeclared=True,
            comparisons=[dict(name="A vs B", a=paths, b=paths)],
        ),
    )
    compare_family(family, tmp_path / "comparison.json", samples=10)
    result = read_json(tmp_path / "comparison.json")["comparisons"][0]
    assert result["delta"] == 0 and result["holm_adjusted_p"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        paired_user_bootstrap([row, row], [row])
    diagnostics(tmp_path, tmp_path / "diagnostics.json")
    assert read_json(tmp_path / "diagnostics.json")["quality_tolerance"] == 0.002


def test_raw_complexity(prepared, csv_path, cfg, tmp_path):
    complexity(csv_path, prepared, cfg, tmp_path / "complexity.json", count=2, warmup=0, repeats=1)
    result = read_json(tmp_path / "complexity.json")
    assert {r["factor"] for r in result["results"]} == {"catalog", "history", "ensemble", "acquisitions"}
    assert all(r["repetitions"][0]["p50_ms"] > 0 for r in result["results"])


def test_pinned_release_rejects_unrecognized_archive(tmp_path):
    raw = tmp_path / "not-pinned.zip"
    raw.write_bytes(b"not data")
    with pytest.raises(ValueError, match="pinned"):
        convert(raw, tmp_path / "bad.csv", "getnext-gowalla")


def test_full_validation_calibration(csv_path, cfg, tmp_path):
    from drlpoi.experiments import calibrate
    import yaml

    cfg = dict(
        cfg,
        predictor_search=[dict(dim=8, layers=1, learning_rate=0.001)],
        controller_search=[
            dict(q_hidden=8, q_lr=0.001, replay_capacity=32, target_interval=2, lambda_cost=0.03)
        ],
        entropy_search=[0.0, 1.0],
    )
    out = tmp_path / "selection"
    frozen = tmp_path / "frozen.yaml"
    calibrate(csv_path, cfg, out, frozen)
    chosen = yaml.safe_load(frozen.read_text())
    assert len(chosen["costs"]) == 4 and chosen["lambda_cost"] == 0.03
    assert chosen["cost_profile_data_identity"] == Store(csv_path, cfg).identity
    assert not (out / "cache/test").exists()
    assert read_json(str(frozen) + ".selection.json")["test_evaluated"] is False


def test_frequency_timing(csv_path, cfg, tmp_path):
    from drlpoi.baselines import frequency_latency, popularity

    for personal in (False, True):
        path = tmp_path / ("personal.json" if personal else "popular.json")
        frequency_latency(csv_path, cfg, path, personal, count=2, warmup=1, repeats=3)
        record = read_json(path)
        assert len(record["repetitions"]) == 3 and record["parameters"] == 0
        _, summary = popularity(csv_path, cfg, personal)
        assert summary["out_of_catalog"] == 1
