"""Regressions from the independent protocol/execution audit (synthetic data)."""

import numpy as np
import pytest
import torch

from drlpoi.cache import Cache, build
from drlpoi.data import Store
from drlpoi.model import Predictor, pack
from drlpoi.predictor import fit, load_ensemble
from drlpoi.runtime import Inference
from drlpoi.util import read_json, write_json


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA execution requires GPU hardware"
            ),
        ),
    ],
)
def test_half_precision_attention_forward_and_backward(csv_path, cfg, selected, device):
    st = Store(csv_path, cfg)
    rs = st.split("fit", True)[:2]
    model = Predictor(len(st.catalog), len(st.cmap), cfg).to(device)
    batch = pack(st, rs, np.full((len(rs), len(st.sources)), selected), device)
    # CPU autocast reproduces the FP16 mask overflow without claiming CUDA validation.
    with torch.autocast(device, dtype=torch.float16):
        logits = model.logits(model(batch))
        loss = torch.nn.functional.cross_entropy(
            logits.float(), torch.tensor([st.target(r) for r in rs], device=device)
        )
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_failed_fit_preserves_metadata_and_checkpoint(csv_path, cfg, tmp_path):
    pred = tmp_path / "pred"
    fit(csv_path, cfg, pred)
    before = {p.name: p.read_bytes() for p in pred.iterdir() if p.is_file()}
    changed = dict(cfg, seed=29)
    with pytest.raises(ValueError, match="Resume configuration/data mismatch"):
        fit(csv_path, changed, pred, resume=True)
    with pytest.raises(FileExistsError):
        fit(csv_path, changed, pred)
    assert before == {p.name: p.read_bytes() for p in pred.iterdir() if p.is_file()}
    load_ensemble(csv_path, pred, "cpu")
    # A same-shaped configuration edit must not silently relabel existing weights.
    write_json(pred / "config.json", changed)
    with pytest.raises(ValueError, match="checkpoint/configuration mismatch"):
        load_ensemble(csv_path, pred, "cpu")


def test_source_order_and_missing_policy_are_rejected(csv_path, cfg, tmp_path):
    pred, cache = tmp_path / "pred", tmp_path / "test"
    fit(csv_path, cfg, pred)
    build(csv_path, pred, cache, "test", "cpu")
    reordered = dict(cfg, sources=list(reversed(cfg["sources"])))
    with pytest.raises(ValueError, match="source order"):
        Cache(cache, reordered)
    with pytest.raises(ValueError, match="source order"):
        Inference(csv_path, pred, reordered)
    engine = Inference(csv_path, pred, cfg)
    with pytest.raises(ValueError, match="policy checkpoint"):
        engine.run(engine.store.split("test")[0], "direct")


def test_worker_tolerates_finished_lock_disappearing(tmp_path, monkeypatch):
    import drlpoi.experiments as ex
    from drlpoi.util import digest
    from types import SimpleNamespace

    pp = tmp_path / "plan.json"
    job = dict(id="a", kind="cpu", dependencies=[], argv=["audit"])
    write_json(pp, dict(jobs=[job]))
    state = tmp_path / "state"
    state.mkdir()
    vanished = state / "finished.lock"
    write_json(vanished, dict(kind="gpu"))
    original = ex.read_json

    def release_then_read(path):
        if path == vanished:
            vanished.unlink()
        return original(path)

    monkeypatch.setattr(ex, "read_json", release_then_read)
    monkeypatch.setattr(ex.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    ex.worker(pp, cpu=True)
    assert read_json(state / (digest(job["id"])[:20] + ".done.json"))["returncode"] == 0


def test_worker_rejects_changed_frozen_costs(tmp_path):
    from drlpoi.experiments import worker
    from drlpoi.util import file_hash

    costs = tmp_path / "reference_costs.json"
    write_json(costs, dict(costs=[1, 2]))
    pp = tmp_path / "plan.json"
    write_json(pp, dict(jobs=[], protocol_hashes={str(costs): file_hash(costs)}))
    write_json(costs, dict(costs=[2, 1]))
    with pytest.raises(ValueError, match="protocol/source changed"):
        worker(pp, cpu=True)


def test_predictor_validation_keeps_out_of_catalog_requests(csv_path, cfg, tmp_path, monkeypatch):
    from dataclasses import replace
    from drlpoi.util import load_torch

    original = Store.split

    def ooc_validation(st, split, eligible=False):
        if split == "val":
            rows = [replace(r, target=len(st.p) - 1) for r in original(st, split)]
            return [] if eligible else rows
        return original(st, split, eligible)

    monkeypatch.setattr(Store, "split", ooc_validation)
    fit(csv_path, cfg, tmp_path / "pred")
    assert load_torch(tmp_path / "pred/member_0.pt")["val_ndcg"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA resume requires GPU hardware")
def test_cuda_amp_checkpoint_resume(csv_path, cfg, tmp_path, monkeypatch):
    import drlpoi.predictor as predictor

    gpu_cfg = dict(cfg, device="cuda", amp=True, ensemble=1, epochs=2)
    original = predictor.save_torch

    def stop_after_checkpoint(path, obj):
        original(path, obj)
        if path.name.endswith(".last.pt") and obj["epoch"] == 0:
            raise RuntimeError("interrupted after CUDA checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(predictor, "save_torch", stop_after_checkpoint)
        with pytest.raises(RuntimeError, match="interrupted after CUDA checkpoint"):
            fit(csv_path, gpu_cfg, tmp_path / "gpu")
    fit(csv_path, gpu_cfg, tmp_path / "gpu", resume=True)
    _, models, _ = load_ensemble(csv_path, tmp_path / "gpu", "cuda")
    assert len(models) == 1
