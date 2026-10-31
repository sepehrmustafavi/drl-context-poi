import torch

from drlpoi.util import environment


def test_environment_omits_device_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def reject_model_query(*args, **kwargs):
        raise AssertionError("Device model must not be queried")

    monkeypatch.setattr(torch.cuda, "get_device_name", reject_model_query)
    record = environment()
    assert record["cuda_available"] is True
    assert "gpu" not in record
