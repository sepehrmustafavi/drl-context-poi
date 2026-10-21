import yaml
from drlpoi.cli import main
from drlpoi.util import read_json


def test_public_cli_path(csv_path, cfg, tmp_path):
    c = dict(cfg, ensemble=1, costs=None, predictor_search=[dict(dim=8, layers=1, learning_rate=0.001)])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(c))
    canonical = tmp_path / "canonical.csv"
    main(["convert", "--format", "canonical", "--raw", str(csv_path), "--out", str(canonical)])
    common = ["--config", str(config), "--csv", str(canonical)]
    pred = tmp_path / "pred"
    cost = tmp_path / "cost.json"
    main(["fit", *common, "--out", str(pred)])
    main(["profile-cost", *common, "--predictor", str(pred), "--out", str(cost), "--count", "4"])
    ca = {s: tmp_path / s for s in ("policy", "val", "test")}
    for s, path in ca.items():
        main(["cache", *common, "--predictor", str(pred), "--split", s, "--out", str(path)])
    p = tmp_path / "controller"
    main(
        [
            "train-policy",
            "--config",
            str(config),
            "--cost-profile",
            str(cost),
            "--train-cache",
            str(ca["policy"]),
            "--val-cache",
            str(ca["val"]),
            "--method",
            "ddqn",
            "--out",
            str(p),
            "--resume",
        ]
    )
    main(
        [
            "evaluate",
            "--config",
            str(config),
            "--cost-profile",
            str(cost),
            "--cache",
            str(ca["test"]),
            "--policy",
            str(p / "policy.pt"),
            "--out",
            str(tmp_path / "evaluation"),
        ]
    )
    main(
        [
            "infer",
            *common,
            "--predictor",
            str(pred),
            "--cost-profile",
            str(cost),
            "--policy",
            str(p / "policy.pt"),
            "--out",
            str(tmp_path / "raw"),
        ]
    )
    a = read_json(tmp_path / "evaluation/summary.json")
    b = read_json(tmp_path / "raw/summary.json")
    assert a["all"] == b["all"]
    main(
        [
            "latency",
            *common,
            "--predictor",
            str(pred),
            "--cost-profile",
            str(cost),
            "--method",
            "fixed",
            "--fixed-mask",
            "1",
            "--count",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--out",
            str(tmp_path / "latency.json"),
        ]
    )
    main(["hpo", *common, "--out", str(tmp_path / "hpo")])
    assert (tmp_path / "hpo/selected.yaml").exists()
