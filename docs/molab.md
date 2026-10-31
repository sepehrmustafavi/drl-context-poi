# Molab execution guide

Use a CUDA-enabled Python 3.11+ runtime. Keep the repository and its artifacts on storage that persists across sessions. Run one worker on device 0 and keep other device workloads stopped for calibration and timing. Runtime metadata records CUDA availability and software versions without recording the device model.

## Setup

In a terminal, from your chosen persistent working directory:

```bash
git clone https://github.com/sepehrmustafavi/drl-context-poi.git
cd drl-context-poi
python -m pip install -e '.[dev,plots,baselines]'
python scripts/check_environment.py
```

Use the runtime's CUDA-enabled PyTorch installation. The environment check includes a small CUDA tensor operation. The experiment launcher checks CUDA availability before starting work.

## Prepare data

Obtain the NYC and Tokyo raw files from [the authors' dataset page](https://sites.google.com/site/yangdingqi/home/foursquare-dataset), extract them and place them at the paths below. See [dataset definitions](data.md).

```bash
mkdir -p data/raw
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_NYC.txt --out data/canonical/nyc.csv
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_TKY.txt --out data/canonical/tky.csv
python scripts/download.py --url https://raw.githubusercontent.com/songyangco/GETNext/6a79081aec6951ebec31bff6400d6ba0e40a613e/dataset/Gowalla-CA.zip --sha256 c5c0604ab2ee68219338541bc23fe7e9f2023c6e9569bafda68488bde87afc3f --out data/raw/Gowalla-CA.zip
python -m drlpoi.cli convert --format getnext-gowalla --raw data/raw/Gowalla-CA.zip --out data/canonical/gowalla_ca.csv
```

## Start the full suite

```bash
set -o pipefail
bash scripts/run_full.sh 2>&1 | tee execution.log
```

The launcher fetches pinned baseline sources, runs tests and data audits, calibrates each dataset on validation, freezes settings, generates the full plan, runs one worker, performs isolated timing and writes reports, paired comparisons, diagnostics and figures. Run this launcher only once at a time. It stops on errors. This starts the full protocol, including the three main seeds 11/29/47.

Alternatively, upload/open [molab_one_gpu.ipynb](../notebooks/molab_one_gpu.ipynb) in the notebook interface. Set `REPO_DIR` in the first code cell to the persistent repository directory. Run setup, prepare the canonical files, then run calibration, worker, timing and reports in order. The worker cell waits for completion.

## Resume and outputs

Preserve `data/`, `selection/`, `configs/frozen/`, `runs/`, `experiments/` and `artifacts/`. Reuse the same checkout, configs and plan when rerunning the launcher. Existing frozen configs and plan files are reused; checkpoint/cache resume follows the standard protocol. Do not update source code during an experiment.

After an interrupted session, confirm that the previous process has stopped before removing a stale `.lock` or `scheduler.mutex`. Inspect and fix a failed job before removing its `.failed.json`. Starting a background process does not extend a provider session's lifetime.

Final report files are in `artifacts/report/`, figures in `artifacts/figures/`, and raw job outputs in `runs/`. Preserve the selection and plan files alongside the reports.
