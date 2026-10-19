# Repository map

| Path | Responsibility |
|---|---|
| `src/drlpoi/data.py` | Converters, shared immutable-length prefix views, fitting-only vocabularies, temporal edge index, bounded retrieval and audit |
| `src/drlpoi/model.py` | Event encoders, Transformer/GRU, source encoders, attention fusion and ensemble prediction |
| `src/drlpoi/predictor.py` | Independent member fitting, validation selection, frozen loading and epoch resume |
| `src/drlpoi/cache.py` | Resumable subset lattice, model/data hashes, reward separation, feasible actions and observable state |
| `src/drlpoi/policy.py` | Q networks, Double DQN/DQN, direct/bandit/myopic policies, batched cached evaluation, replay and resume |
| `src/drlpoi/runtime.py` | Raw lazy retrieval/inference, fitting-only source profiling and GPU-synchronized latency |
| `src/drlpoi/baselines.py` | Global popularity and personal frequency references |
| `src/drlpoi/metrics.py` | Exact full-catalog rank, all/in-catalog metrics, user/request macro, paired cluster statistics |
| `src/drlpoi/analysis.py` | Training-defined history groups and Holm p-value adjustment |
| `src/drlpoi/external.py` | Official baseline split/catalog export and strict full-score import; not upstream training adapters |
| `src/drlpoi/experiments.py` | Main/supplementary experiment DAG, config/data fingerprints, exclusive worker locks and bounded HPO |
| `src/drlpoi/report.py` | Completed-run CSV, seed mean/sample standard deviation, Markdown tables |
| `src/drlpoi/figures.py` | PNG/PDF figures from actual aggregate results |
| `src/drlpoi/cli.py` | Public command line |
| `src/drlpoi/util.py` | Atomic file writes, trusted checkpoint loading, hashing, devices, seeds and environment capture |
| `configs/base.yaml` | Declared starting settings; freeze validation-selected settings separately |
| `configs/upstream.json` | Pinned official repository commits and adapter status |
| `scripts/` | Data download, pinned-source checkout, environment check, manual GitHub push |
| `notebooks/` | Kaggle dual-worker and Colab single-worker entry points |
| `tests/` | Public CLI, complete internal pipeline, temporal leakage, constraints, replay resume, all backbones, experiment DAG and metrics |
| `docs/chapter4.md` | Mapping to the nine evaluation sections and asymptotic analysis |
| `.github/workflows/tests.yml` | CPU automated tests and static undefined-name/import checks |

Runtime artifacts are intentionally ignored by Git: canonical/raw data, weights, subset caches, upstream checkouts, generated figures and large result files. Preserve them outside Git and publish permitted checksums, configurations and aggregate results separately.
