from pathlib import Path
import pandas as pd
import pytest
import yaml


@pytest.fixture
def cfg():
    c = yaml.safe_load(Path("configs/base.yaml").read_text())
    c.update(
        device="cpu",
        threads=1,
        dim=8,
        heads=2,
        layers=1,
        dropout=0.0,
        ensemble=2,
        recent=3,
        block_size=3,
        min_history=2,
        batch_size=16,
        epochs=1,
        patience=2,
        policy_epochs=1,
        policy_steps_per_epoch=16,
        policy_batch_size=4,
        replay_warmup=4,
        q_hidden=8,
        target_interval=2,
        costs=[1.0, 1.0, 1.0, 1.0],
        amp=False,
    )
    return c


@pytest.fixture
def csv_path(tmp_path):
    # Synthetic fixture ONLY; never used as a publication dataset.
    rows = []
    start = pd.Timestamp("2020-01-01", tz="UTC")
    for n in range(36):
        for u in range(3):
            p = (n + u) % 7
            rows.append(
                dict(
                    user_id=str(u),
                    poi_id=str(p),
                    category=str(p % 2),
                    timestamp=(start + pd.Timedelta(hours=6 * n, minutes=u)).isoformat(),
                    latitude=35 + p * 0.001,
                    longitude=139 + p * 0.001,
                )
            )
    # One future-only POI must remain outside the fitting catalog.
    rows[-1]["poi_id"] = "future-only"
    path = tmp_path / "events.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
