# DRL Context POI

Sequential context acquisition for next-POI recommendation: a frozen contextual predictor ensemble supplies scores and state; Double DQN chooses context sources and STOP under a fixed budget.

The Chapter 4 experiment matrix is executable: three datasets, seeds 11/29/47, internal acquisition policies, ablations, sensitivity, pinned official-model adapters, and a separate timing phase. This is **implementation coverage**, not completed scientific evaluation. No trained real-data recommendation results are included. Start with [the audit](docs/AUDIT_FA.md), [coverage](docs/coverage.md) and [baseline adaptations](docs/external.md). The [2026-10-21 re-audit](docs/re-audit-2026-10-21.md) records execution defects fixed in 0.2.1 and remaining GPU validation.

## Installation

Use Python 3.11+ and install PyTorch for your CUDA driver using the [official selector](https://pytorch.org/get-started/locally/). T4 workers train independent jobs; their memory is not pooled.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the appropriate PyTorch CUDA wheel here.
python -m pip install -e '.[dev,plots,baselines]'
python scripts/check_environment.py
python -m pytest -q
```

Fetch pinned sources before running disconnected workers:

```bash
for MODEL in sasrec getnext dchl ipcm; do
  python scripts/fetch_upstream.py "$MODEL"
done
DRLPOI_UPSTREAM_ROOT="$PWD/artifacts/upstream" python -m pytest tests/test_upstream.py -q
```

Upstream model classes are loaded directly from verified commits; stock training entry points are not executed. The adapters rebuild training graphs, retain cold users and out-of-catalog targets, and select checkpoints on validation. See the material protocol adaptations in [docs/external.md](docs/external.md). Synthetic tests are software checks and must not become article results.

## Data

[Data instructions](docs/data.md) specify the original Foursquare format and exact Gowalla release. Use the complete chosen release, with no pilot sampling.

```bash
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_NYC.txt --out data/canonical/nyc.csv
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_TKY.txt --out data/canonical/tky.csv
python scripts/download.py --url https://raw.githubusercontent.com/songyangco/GETNext/6a79081aec6951ebec31bff6400d6ba0e40a613e/dataset/Gowalla-CA.zip --sha256 c5c0604ab2ee68219338541bc23fe7e9f2023c6e9569bafda68488bde87afc3f --out data/raw/Gowalla-CA.zip
python -m drlpoi.cli convert --format getnext-gowalla --raw data/raw/Gowalla-CA.zip --out data/canonical/gowalla_ca.csv
python -m drlpoi.cli audit --csv data/canonical/nyc.csv --out artifacts/nyc_audit.json
python -m drlpoi.cli audit --csv data/canonical/tky.csv --out artifacts/tky_audit.json
python -m drlpoi.cli audit --config configs/gowalla.yaml --csv data/canonical/gowalla_ca.csv --out artifacts/gowalla_ca_audit.json
```

The pinned Gowalla archive's three splits are merged and split again using the required global 50/20/10/20 **time-range fractions**. Upstream graph files are discarded. Categories are conservatively disabled. The inherited release filtering is documented; this is not a new extraction of every California check-in from SNAP. A SNAP rectangle variant remains supported under its own name.

## Select and freeze configurations

Run this phase serially on the idle reference GPU 0. It uses all fitting data and validation, seed 11, and never generates a test cache or evaluates test performance. The bounded predictor search includes dimension, depth and learning rate; controller search includes Q width, learning rate, replay capacity, target update and cost weight. Entropy threshold is selected on validation. The main budget fraction and acquisition cap are declared in the input config; defaults are .5 and J.

```bash
CUDA_VISIBLE_DEVICES=0 python -m drlpoi.cli calibrate --csv data/canonical/nyc.csv --out selection/nyc --export configs/frozen/nyc.yaml
CUDA_VISIBLE_DEVICES=0 python -m drlpoi.cli calibrate --csv data/canonical/tky.csv --out selection/tky --export configs/frozen/tky.yaml
CUDA_VISIBLE_DEVICES=0 python -m drlpoi.cli calibrate --config configs/gowalla.yaml --csv data/canonical/gowalla_ca.csv --out selection/gowalla_ca --export configs/frozen/gowalla_ca.yaml
```

Declare custom finite searches in YAML using `predictor_search`, `controller_search`, and `entropy_search` **before** running. [The protocol](docs/protocol.md) gives defaults. Search records, checkpoints and the measured reference cost profile are retained in `selection/`; preserve them with `configs/frozen/`. No candidate is shortened to a pilot dataset. Calibration and final fitting can take substantial time; defaults are not claimed to fit a particular notebook quota.

Lower-level `hpo` and `controller-hpo` commands are also available. `calibrate` resumes predictor/controller trials and completed cache records; cost profiling is reused only when its predictor hashes match.

## Generate and run Chapter 4

```bash
python -m drlpoi.cli plan --suite full --settings-dir configs/frozen --out experiments/final
```

Inspect `plan.json` and the predeclared `family_*.json` before execution. The plan records code/config/data hashes, hashes frozen reference costs, comparison families and upstream pins, and declares all sensitivity values explicitly. Run these in two terminals on one shared machine/filesystem:

```bash
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 0
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 1
```

Workers claim independent jobs. Reference profiling is serialized against other jobs and belongs to GPU 0. On a single GPU use only the first command. When all training/evaluation jobs have completed, stop other GPU jobs and run:

```bash
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 0 --timing
```

Timing cannot start before the non-timing phase completes and uses only GPU 0. Default measurements: 50 warmups, up to 1000 requests, three repeats on the same checkpoint, GPU synchronization. Raw retrieval, descriptor access, base encoding, cached source encoding, fusion, full-catalog scoring and policy decisions are included. Timing repeats are not training seeds. Acquisition costs and latency are separate quantities.

The `complexity` jobs additionally vary active catalog scoring size, history length, acquisitions and ensemble size on trained weights under a controlled fixed policy. These are workload measurements, not accuracy measurements for truncated catalogs. All checkpoint weights remain resident in this sweep; deployment memory for M=1/M=3 is separately available from their latency jobs.

Keep `selection/`, `runs/`, `experiments/`, canonical data and frozen configs across notebook shutdowns. Resume training at completed epochs; caches resume at committed requests. After a crash, remove a stale `.lock` or `scheduler.mutex` only after verifying its worker has stopped. Fix a failed job before removing its `.failed.json`. Do not share scheduler state between independent machines through an eventually synchronized Drive folder.

## Reports and statistics

```bash
python -m drlpoi.cli report --root runs --out artifacts/report
for DATASET in nyc tky gowalla_ca; do
  python -m drlpoi.cli compare-family --family "experiments/final/family_$DATASET.json" --out "artifacts/report/paired_$DATASET.json"
done
python -m drlpoi.cli diagnostics --root runs --out artifacts/report/discussions.json
python -m drlpoi.cli figures --aggregate artifacts/report/aggregate.json --out artifacts/figures
```

Reports include request/user macro metrics for all/in-catalog populations, in-catalog NLL, acquisition costs/counts, out-of-catalog counts and available timing summaries. Main results use sample standard deviation across 11/29/47; single-seed results have no invented standard deviation. Paired families average matched seeds per request, bootstrap users and apply Holm within each predeclared dataset family. Between-seed variation is reported separately.

Discussion diagnostics cover seed-11 NYC/Tokyo quality–cost frontiers, .002 absolute NDCG@10 tolerance, acquisition groups and reproducibly selected successful/unsuccessful traces. These are descriptive test analyses, not a way to tune deployment settings. The offline label-dependent oracle minimizes final NLL plus cost on in-catalog requests; it is neither deployable nor a guaranteed NDCG upper bound.

## Individual commands

`python -m drlpoi.cli --help` lists the data, fitting, caching, policy training, raw inference, profiling, baseline, HPO, scheduling, statistics and reporting commands. [Chapter 4 mapping](docs/chapter4.md) identifies outputs for every section.

Version 0.2 changes the predictor architecture and temporal protocol identity. Rebuild checkpoints/caches from version 0.1; they are not compatible. Actual CUDA accuracy, memory and latency must be verified on the final T4 environment. No real-data training result or hardware speed claim is supplied with this repository.
