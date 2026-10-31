#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export DRLPOI_UPSTREAM_ROOT="$PWD/artifacts/upstream"
for dataset in nyc tky gowalla_ca; do
  test -f "data/canonical/$dataset.csv" || { echo "Prepare data/canonical/$dataset.csv using docs/data.md" >&2; exit 1; }
done
python scripts/check_environment.py
python -c 'import torch; assert torch.cuda.is_available(), "Enable a CUDA runtime"'
for model in sasrec getnext dchl ipcm; do
  python scripts/fetch_upstream.py "$model"
done
python -m pytest -q
for dataset in nyc tky gowalla_ca; do
  config=configs/base.yaml
  if [ "$dataset" = gowalla_ca ]; then config=configs/gowalla.yaml; fi
  python -m drlpoi.cli audit --config "$config" --csv "data/canonical/$dataset.csv" --out "artifacts/${dataset}_audit.json"
  if [ ! -f "configs/frozen/$dataset.yaml" ]; then
    python -m drlpoi.cli calibrate --config "$config" --csv "data/canonical/$dataset.csv" --out "selection/$dataset" --export "configs/frozen/$dataset.yaml"
  fi
done
if [ ! -f experiments/final/plan.json ]; then
  python -m drlpoi.cli plan --suite full --settings-dir configs/frozen --out experiments/final
fi
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 0
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 0 --timing
python -m drlpoi.cli report --root runs --out artifacts/report
for dataset in nyc tky gowalla_ca; do
  python -m drlpoi.cli compare-family --family "experiments/final/family_$dataset.json" --out "artifacts/report/paired_$dataset.json"
done
python -m drlpoi.cli diagnostics --root runs --out artifacts/report/discussions.json
python -m drlpoi.cli figures --aggregate artifacts/report/aggregate.json --out artifacts/figures
