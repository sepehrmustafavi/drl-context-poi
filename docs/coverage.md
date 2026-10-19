# Implementation coverage

| Component | Status |
|---|---|
| Raw Foursquare/Gowalla/canonical conversion and SHA-256 manifests | Implemented |
| Temporal requests, fitting-only catalog, historical graph cutoff | Implemented and tested |
| Causal spatiotemporal Transformer / contextual GRU / independent ensemble | Implemented |
| Bounded remote, periodic, transition, category sources | Implemented; category disabled for Gowalla in generated plan |
| Double DQN, DQN, direct subset selector, bandit, myopic | Implemented and CPU end-to-end tested |
| STOP, masks, budget, finite horizon and undiscounted terminal NLL | Implemented and tested |
| Internal reference policies and target-dependent oracle diagnostic | Implemented |
| Predictor/controller resume and resumable subset lattice caches | Implemented; resume occurs at completed epoch/request boundaries |
| Main/supplementary experiment DAG; independent GPU workers | Implemented; CPU-worker and DAG tests |
| HPO | Bounded predictor HPO implemented; full controller HPO not automatically selected |
| Raw inference and latency/profiling | Implemented; CPU exercised; real CUDA measurements pending |
| Metrics, aggregate reports, user-cluster bootstrap/sign-flip, Holm function | Implemented |
| Original SASRec reproduction | Not claimed; local baseline is SASRec-style |
| GETNext/DCHL/iPCM training adapters | **Not implemented**; pinned fetch/export/import contracts available |
| HRL-PRP / ROTAN | Not implemented; optional external additions, no results claimed |
| Real-data benchmark runs and GPU timing | **Not run in this delivery** |
| Online causal/user-utility evaluation | Outside the supported offline-check-in task |

A repository can be runnable while a publication's external evaluation is unfinished. Do not describe this package as having reproduced the external models or completed the article's experiments. The numerical outputs of unit tests are fixture diagnostics and must not be copied into article tables.
