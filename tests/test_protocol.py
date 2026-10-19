import numpy as np
import pandas as pd
import torch
import pytest
from drlpoi.data import Store
from drlpoi.model import Predictor, pack, ensemble_predict
from drlpoi.metrics import rank_of, per_request, paired_user_bootstrap
from drlpoi.policy import td_targets


def test_catalog_and_prefix_time(csv_path, cfg):
    st = Store(csv_path, cfg)
    assert "future-only" not in st.catalog
    for r in st.requests:
        assert max(r.prefix) < r.target
        assert all(st.t[i] <= st.t[r.prefix[-1]] for i in r.prefix)
        for j in range(len(st.sources)):
            ids = st.retrieve(r, j)
            assert len(ids) <= cfg["block_size"]
            assert all(st.t[i] <= st.t[r.prefix[-1]] for i in ids)
            if st.sources[j] in ("transition", "category"):
                assert all(st.t[i] < min(st.t[r.prefix[-1]], st.cuts[0]) for i in ids)
    assert any(st.target(r) == -1 for r in st.split("test"))


def test_future_labels_do_not_change_observations(csv_path, cfg, tmp_path):
    st = Store(csv_path, cfg)
    r = st.split("test")[0]
    df = pd.read_csv(csv_path, dtype={"user_id": str, "poi_id": str, "category": str})
    # Preserve target time and request prefix; change only the future target's POI/category/coords.
    df.loc[r.target, ["poi_id", "category", "latitude", "longitude"]] = ["invented", "999", 80, 10]
    path = tmp_path / "mutated.csv"
    df.to_csv(path, index=False)
    other = Store(path, cfg)
    rr = next(x for x in other.requests if x.id == r.id)
    assert np.array_equal(st.descriptors(r)[1], other.descriptors(rr)[1])
    a = pack(st, [r], np.ones((1, 4), bool), "cpu")
    b = pack(other, [rr], np.ones((1, 4), bool), "cpu")
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_lazy_retrieval_and_empty_fusion(csv_path, cfg, monkeypatch):
    st = Store(csv_path, cfg)
    r = st.split("test")[0]
    calls = []
    original = st.retrieve

    def retrieve(r, j):
        calls.append(j)
        return original(r, j)

    monkeypatch.setattr(st, "retrieve", retrieve)
    empty = pack(st, [r], np.zeros((1, 4), bool), "cpu")
    assert calls == []
    selected = pack(st, [r], np.array([[1, 0, 0, 0]], bool), "cpu")
    assert calls == [0]
    model = Predictor(len(st.catalog), len(st.cmap), cfg).eval()
    z = model(empty)
    # Mutating unacquired blocks cannot influence empty-context output.
    empty["sp"] = selected["sp"]
    empty["sc"] = selected["sc"]
    empty["sf"] = selected["sf"]
    assert torch.allclose(z, model(empty))
    p, z, h, u = ensemble_predict([model], empty)
    assert torch.allclose(p.sum(1), torch.ones(1))
    assert torch.allclose(u, torch.zeros_like(u), atol=1e-6)


def test_ddqn_selection_and_terminal_target():
    online = torch.tensor([[1.0, 9.0, 100.0], [1.0, 9.0, 100.0]])
    target = torch.tensor([[8.0, 2.0, 1000.0], [8.0, 2.0, 1000.0]])
    valid = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    rewards = torch.tensor([-1.0, -3.0])
    done = torch.tensor([False, True])
    assert torch.equal(td_targets(rewards, done, online, target, valid, True), torch.tensor([1.0, -3.0]))
    assert torch.equal(td_targets(rewards, done, online, target, valid, False), torch.tensor([7.0, -3.0]))


def test_full_rank_ties_and_ooc():
    assert rank_of([0.2, 0.4, 0.4], 2) == 2
    assert rank_of([0.2, 0.4, 0.4], 1) == 1
    assert rank_of([0.2, 0.4, 0.4], -1) is None
    assert per_request(None)["ndcg@10"] == 0


def test_clustered_bootstrap():
    a = [dict(request_id=str(i), user=str(i // 2), in_catalog=True, **{"ndcg@10": 1.0}) for i in range(8)]
    b = [dict(x, **{"ndcg@10": 0.0}) for x in a]
    result = paired_user_bootstrap(a, b, samples=50)
    assert result["delta"] == 1 and result["ci95"] == [1.0, 1.0]


@pytest.mark.parametrize("backbone", ["gru", "sasrec"])
def test_backbone_forward_and_backward(csv_path, cfg, backbone):
    c = dict(cfg, backbone=backbone)
    st = Store(csv_path, c)
    model = Predictor(len(st.catalog), len(st.cmap), c)
    rs = st.split("fit", True)[:2]
    batch = pack(st, rs, np.ones((len(rs), 4), bool), "cpu")
    logits = model.logits(model(batch))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([st.target(r) for r in rs]))
    loss.backward()
    assert torch.isfinite(loss) and model.poi.weight.grad is not None
