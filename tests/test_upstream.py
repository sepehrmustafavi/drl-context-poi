"""Opt-in integration tests execute the actual pinned upstream classes."""

import os
import numpy as np
import pytest
import torch
from drlpoi.upstream import Adapted, fetch, load_module, baseline
from drlpoi.data import Store
from drlpoi.util import read_json, seed_everything

ROOT = os.getenv("DRLPOI_UPSTREAM_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="Set DRLPOI_UPSTREAM_ROOT to pinned source directory")


@pytest.mark.parametrize("name", ["sasrec", "getnext", "dchl", "ipcm"])
def test_official_adapter(csv_path, cfg, tmp_path, name):
    cfg = dict(cfg, dropout=0.2)
    seed_everything(11, 1)
    from drlpoi.upstream import ROOT as PACKAGE

    path, _ = fetch(name, ROOT)
    meta = read_json(PACKAGE / "configs/upstream.json")[name]
    st = Store(csv_path, cfg)
    net = Adapted(
        st, cfg, name, load_module(name, path / meta.get("module", "model.py")), torch.device("cpu")
    )
    net.train()
    loss = net.loss(st.split("fit", True)[:3])
    assert torch.isfinite(loss)
    loss.backward()
    net.eval()
    requests = st.split("test")[:3]
    p = net.probabilities(requests)
    assert p.shape == (3, len(st.catalog))
    assert torch.allclose(p.sum(-1), torch.ones(3))
    assert torch.allclose(p, net.probabilities(requests)), "Evaluation must be deterministic"
    from dataclasses import replace

    changed = [replace(r, target=0, user="unseen-user") for r in requests]
    unknown = net.probabilities(changed)
    assert torch.isfinite(unknown).all()
    assert torch.allclose(p, net.probabilities([replace(r, target=0) for r in requests]))
    # Future-only targets are retained by the complete training/evaluation entry point.
    summary = baseline(csv_path, cfg, name, tmp_path / name, ROOT)
    assert summary["out_of_catalog"] == 1
    assert summary["requests"] == len(st.split("test"))
    assert np.isfinite(summary["nll_in_catalog"])
