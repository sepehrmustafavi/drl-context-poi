# Implementation coverage

Audited against the supplied Chapter 3 and the requested Chapter 4 protocol, starting from upstream commit `7c7df21`. The table maps protocol requirements to implementation and software checks.

| Requirement | Implementation | Checks / protocol |
|---|---|---|
| Global elapsed-time 50/20/10/20 split, fit catalog, no future target inputs | `data.Store`, feature-only prediction paths | Leakage, UTC units, strict request order and OOC tests |
| Exact dataset versions and partition audit | Raw converters, pinned Gowalla-CA converter, expanded `audit` | Actual released Gowalla metadata in `gowalla-ca-audit.json`; supply raw NYC/Tokyo and run audits |
| Small causal Transformer, M=3, matched GRU | `model`, `predictor`, final plan | CPU pipeline tests; shared dataset protocol |
| Chapter 3 query/output projections and lazy request caching | Learned Q/output projections, `RequestPrediction` | Output equivalence and one-encoding-per-block regression test |
| Three main seeds and frozen predictor per policy comparison | Immutable DAG, cache/checkpoint hashes | DAG and immutable configuration tests |
| All/request and in-catalog/request plus equal-user metrics | `metrics`, `report` | Full-rank ties, OOC misses, NLL, aggregation tests |
| Bounded predictor/controller/entropy HPO | `hpo`, `controller-hpo`, `calibrate` | Full selection-path test; no test cache in calibration |
| Global popularity and personal frequency | Internal deterministic baselines | One run per dataset, no manufactured SD |
| SASRec | Pinned maintained upstream architecture, sampled BCE adaptation | Actual upstream-class CPU training/inference test; protocol differences documented |
| GETNext / DCHL / iPCM | Pinned upstream-class common-protocol adapters | All four adapters trained/evaluated on synthetic fixtures |
| DDQN, DQN, random-valid, fixed, entropy, direct, myopic, subset bandit | `policy`, frozen shared predictor | End-to-end tests; all feasible subsets include empty/multisource |
| Offline oracle | Minimum terminal NLL + lambda-cost, in-catalog diagnostic | No deployment/NDCG-upper-bound claim |
| Base/all on every dataset and seed | Main DAG references | All-context is unconstrained |
| No STOP / no uncertainty / no representation | Retrained controllers, all datasets, seed 11 | Complete CPU tests |
| Source removal, M=1 vs M=3, GRU vs Transformer | NYC/Tokyo seed-11 DAG | Predictor reused for policy ablations; separate fits for architecture changes |
| Budget 0/.25/.5/.75/1, lambda 0/.03/.1/.3/1, cap 1..J | Every value explicit, NYC/Tokyo seed 11 | Every controller retrained, no inference-only change |
| End-to-end median/p95, parameter counts, memory | Raw `latency`, `baseline-latency`, `frequency-latency`, separate reference GPU phase | Separate profiling phase on the reference device |
| Scaling in catalog/history/acquisitions/ensemble | Raw controlled-workload `complexity` | CPU execution test; scope and resident-memory limitation recorded |
| User-cluster paired bootstrap and predeclared Holm families | `compare-family`, DAG family files | Matched seeds averaged per request; user resampling; seed SD separate |
| History groups, traces, cost at matched quality, quality/cost frontier | `subgroups`, `diagnostics` | NYC/Tokyo seed 11, tolerance .002; descriptive, not test tuning |
| Independent GPU workers and resume | Serialized claims, reference profiling exclusion, separate timing phase | CPU worker/resume/DAG tests; external processes must be stopped for timing |
| Social context | Disabled: no timestamp-verified friendship snapshot was supplied | No fabricated source |
| HRL-PRP / ROTAN | Optional and not implemented | Not mandatory under the requested protocol |

Software checks exercise execution paths and protocol invariants. Use the reporting commands to aggregate experiment metrics and paired comparisons.
