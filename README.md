# DRL Context POI

Budgeted, sequential acquisition of historical context for next-POI recommendation in location-based social networks.

The agent chooses **which context source to acquire next and when to STOP**. A separately trained, frozen ensemble predicts the next recorded POI. This is reinforcement learning for information acquisition within a recommendation request; it is **not** a learned simulator of the user's future mobility.

**Repository status:** the proposed method, internal policy comparisons, chronological data pipeline, experiment scheduler, metrics and raw latency path are implemented. Automated tests use explicitly synthetic fixtures. No real-data performance results, trained weights or publication claims are bundled. GETNext, DCHL and iPCM have pinned-source fetch tools and a strict common-protocol exchange, **not implemented training adapters**. See [coverage](docs/coverage.md) before describing a baseline as reproduced.

## Layout

```text
drl-context-poi/
├── src/drlpoi/              # Data, models, context acquisition, RL and evaluation
├── configs/                # Hyperparameters and pinned upstream sources
├── scripts/                # Environment, dataset downloads and GitHub commands
├── notebooks/              # Kaggle and Colab entry points
├── tests/                  # Leakage, budget, TD-target and end-to-end tests
├── docs/                   # Protocol, Chapter 4 mapping and baseline status
├── .github/workflows/      # CPU continuous integration
├── pyproject.toml
└── README.md
```

## Install

Use Python 3.10+ (3.11 recommended). Create and activate an environment below. Before installing this package, install the appropriate PyTorch CUDA wheel in that environment using the [official selector](https://pytorch.org/get-started/locally/).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the CUDA PyTorch wheel selected for your driver here.
python -m pip install -e '.[dev]'
python scripts/check_environment.py
python -m pytest -q
```

On Kaggle/Colab, use the preinstalled CUDA PyTorch where compatible; install the package into that same Python environment. CPU execution is supported. The tested local CPU package versions are recorded in `docs/verification.md`; the dependency ranges are not a claim of testing every possible combination.

## Prepare real datasets

See [data instructions](docs/data.md) for source links, schemas and the geographical definition. Raw files are not distributed by this repository.

```bash
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_NYC.txt --out data/canonical/nyc.csv
python -m drlpoi.cli convert --format foursquare --raw data/raw/dataset_TSMC2014_TKY.txt --out data/canonical/tky.csv
python -m drlpoi.cli convert --format gowalla --raw data/raw/loc-gowalla_totalCheckins.txt.gz --bbox 32.5 42.0 -124.5 -114.1 --out data/canonical/gowalla_ca.csv
python -m drlpoi.cli audit --csv data/canonical/nyc.csv --out artifacts/nyc_audit.json
```

The last command creates a **California-region bounding-box variant**, not a canonical published Gowalla-CA release. Name and report it accordingly; a rectangle includes some locations outside the state. If using a published Gowalla-CA release instead, preserve its exact source/checksum and convert its actual schema to canonical CSV. Do not silently equate these variants.

## Run one final configuration

These commands use the complete configured data, not a pilot subset. Defaults are starting hyperparameters; freeze validation-selected settings before publication runs.

```bash
python -m drlpoi.cli fit --csv data/canonical/nyc.csv --config configs/base.yaml --out runs/nyc/transformer/seed11/predictor --resume
python -m drlpoi.cli profile-cost --csv data/canonical/nyc.csv --predictor runs/nyc/transformer/seed11/predictor --out runs/nyc/transformer/reference_costs.json
```

Build three independent caches:

```bash
for SPLIT in policy val test; do
  python -m drlpoi.cli cache --csv data/canonical/nyc.csv --predictor runs/nyc/transformer/seed11/predictor --split "$SPLIT" --out "runs/nyc/transformer/seed11/cache/$SPLIT"
done
```

Train and evaluate Double DQN:

```bash
python -m drlpoi.cli train-policy --train-cache runs/nyc/transformer/seed11/cache/policy --val-cache runs/nyc/transformer/seed11/cache/val --cost-profile runs/nyc/transformer/reference_costs.json --method ddqn --out runs/nyc/transformer/seed11/policies/ddqn --resume
python -m drlpoi.cli evaluate --cache runs/nyc/transformer/seed11/cache/test --cost-profile runs/nyc/transformer/reference_costs.json --policy runs/nyc/transformer/seed11/policies/ddqn/policy.pt --out runs/nyc/transformer/seed11/policies/ddqn/test
```

Use `--method dqn`, `myopic`, `direct`, `bandit`, `no_stop`, `no_uncertainty` or `no_representation` when training another controller. Reference evaluation methods are `base`, `all`, `fixed`, `entropy`, `random` and `oracle`. `fixed` requires `--val-cache`; `oracle` is an offline, in-catalog diagnostic using the hidden target. `all` is an unconstrained quality reference and may exceed the budget.

### HPO and final experiment plan

A bounded predictor HPO is available; it uses the full fit interval and validation only. It retains the configured ensemble size and uses one fixed seed. It is not a separate pilot phase.

```bash
python -m drlpoi.cli hpo --csv data/canonical/nyc.csv --out runs/hpo/nyc --set seed=11
```

Run the equivalent HPO for Tokyo and Gowalla using the latter's three-source settings. Freeze the chosen per-dataset configurations as `configs/frozen/nyc.yaml`, `tky.yaml`, `gowalla_ca.yaml`. Select controller budget, cost weight and entropy threshold using validation, then freeze those too. The HPO command optimizes predictor dimension/learning rate; it does **not** claim to optimize the entire method automatically. See [protocol](docs/protocol.md).

```bash
python -m drlpoi.cli plan --suite full --settings-dir configs/frozen --out experiments/final
```

Omit `--settings-dir` only if deliberately using the predeclared defaults. The plan contains the main three-seed comparisons, single-seed ablations/sensitivity, secondary GRU and single-member tests. It writes executable jobs, not results. View exact jobs in `experiments/final/plan.json`.

### Two T4 GPUs

Run these in two terminals on the **same machine and shared filesystem**:

```bash
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 0
python -m drlpoi.cli worker --plan experiments/final/plan.json --gpu 1
```

Workers claim independent runs using exclusive lock files and satisfy dependencies. They do not combine two 16-GB cards into one 32-GB memory pool. The same commands run on one RTX 3090 or L40S with one worker on GPU 0. Avoid simultaneous latency profiling and training.

Save `runs/`, `experiments/` and the canonical data across notebook shutdowns. Predictor and controller training resume from completed epochs; cache generation resumes from committed requests. After an interrupted worker, confirm that its process is dead before removing its stale `.lock`. After fixing a failed job, remove its `.failed.json` and rerun the worker. Never share this scheduler's lock state across independent Colab machines through an eventually synchronized drive.

## Measure deployment latency

This executes raw retrieval and prediction, not the precomputed cache:

```bash
python -m drlpoi.cli latency --csv data/canonical/nyc.csv --predictor runs/nyc/transformer/seed11/predictor --policy runs/nyc/transformer/seed11/policies/ddqn/policy.pt --cost-profile runs/nyc/transformer/reference_costs.json --out artifacts/nyc_ddqn_latency.json
```

It includes descriptors, base prediction, controller decisions, retrieval, block encoding and prediction refresh. Report the GPU, warmup, request count, parameter count, peak allocated CUDA memory and three timing repetitions. The additive acquisition cost is a training proxy and is reported separately from wall-clock latency.

## Results and statistical comparison

```bash
python -m drlpoi.cli report --root runs --out artifacts/report
python -m drlpoi.cli compare --a runs/nyc/transformer/seed11/policies/ddqn/test/requests.jsonl --b runs/nyc/transformer/seed11/policies/dqn/test/requests.jsonl --out artifacts/ddqn_vs_dqn.json
```

Reports distinguish one run from three runs and omit an undefined one-run standard deviation. The comparison uses paired user-cluster bootstrap confidence intervals and a paired cluster sign-flip Monte Carlo p-value. Predeclare comparisons and apply Holm correction across the comparison family (`drlpoi.analysis.holm`). A single seed's interval does not replace between-seed variability.

## External baselines

```bash
python scripts/fetch_upstream.py dchl
python -m drlpoi.cli export-external --csv data/canonical/nyc.csv --out artifacts/exchange/nyc
```

Follow [the exchange contract](docs/external.md) to adapt official training and inference. Import full-catalog prediction scores with `import-external`. Import validation checks coverage, catalog order, split fingerprints and an explicit audit declaration; it cannot prove that an external author's code used no future information.

## Publish this repository

Review the source and run tests first. Large datasets, checkpoints and results are ignored. GitHub CLI:

```bash
git init
git branch -M main
git add .
git commit -m "Implement budgeted DRL context acquisition for POI recommendation"
gh auth login
gh repo create sepehrmustafavi/drl-context-poi --public --source=. --remote=origin --push
```

Alternatively create an empty repository on GitHub, then:

```bash
git remote add origin https://github.com/sepehrmustafavi/drl-context-poi.git
git push -u origin main
```

The package does not create a GitHub repository or push anything automatically.

## Export figures and history subgroups

```bash
python -m pip install -e '.[plots]'
python -m drlpoi.cli figures --aggregate artifacts/report/aggregate.json --out artifacts/figures
python -m drlpoi.cli subgroups --csv data/canonical/nyc.csv --results runs/nyc/transformer/seed11/policies/ddqn/test/requests.jsonl --out artifacts/nyc_history_groups.json
```

Figures use actual completed results only and export 300-dpi PNG plus vector PDF. A single run is labeled n=1 without a fictitious standard deviation. Caption error bars as between-run sample standard deviations, not confidence intervals.
