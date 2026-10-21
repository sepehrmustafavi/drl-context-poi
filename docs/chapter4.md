# Chapter 4 evaluation mapping

| Section | Commands / files | Outputs to report |
|---|---|---|
| 4.1 Hardware and Software Environment | `check_environment.py`, predictor/policy configs, timing environment | GPU/CPU/RAM, Python/PyTorch/CUDA/cuDNN, precision, seed, settings, commit |
| 4.2 Datasets | `convert`, `audit`, `docs/data.md` | Source/hash/version, UTC bounds, event/POI/user/request counts per partition, history quantiles, sparsity, OOC counts |
| 4.3 Evaluation metrics | `evaluate`, baseline adapters, `report` | Recall@5/10/20, NDCG@10 primary, full-rank MRR; all/in-catalog and request/user macro; in-catalog NLL, costs, acquisition count, latency |
| 4.4 Hyperparameter Optimization | `calibrate`, immutable `selection/*` and `configs/frozen/*` | Six bounded full-data predictor trials, six controller candidates, entropy thresholds; seed 11; validation only; frozen budget/cap/costs |
| 4.5 Overall Performance | Main transformer jobs plus `baseline` jobs | DDQN vs popularity, personal frequency, official-class SASRec/GETNext/DCHL/iPCM adaptations; three main seeds for learned methods |
| 4.6 Ablation Study | Main references and retrained supplementary policies | Base/all: all datasets, 3 seeds. No STOP/uncertainty/representation: all datasets, 1 seed. Source removal/M/GRU: NYC/Tokyo, 1 seed |
| 4.7 Complexity Analysis | `worker --timing`, `latency`, `baseline-latency`, `frequency-latency`, `complexity` | Parameters, GPU peak allocation, process RSS, p50/p95 end-to-end timings, controlled workload scaling; three timing repeats |
| 4.8 Baseline Comparison | Main policy jobs and offline oracle | Random-valid, fixed, entropy, direct subset selector, myopic, subset bandit, DQN, DDQN; same cache/predictor/resources/costs/budget/cap |
| 4.9 Discussions | Explicit sensitivity jobs, `subgroups`, `diagnostics` | NYC/Tokyo seed 11; budget 0/.25/.5/.75/1, lambda 0/.03/.1/.3/1, cap 1..J; traces, history, Pareto frontier, .002 quality matching |

`family_{dataset}.json` is emitted before any result analysis. The primary family comprises DDQN against nine internal policies/references, four learned recommenders and two deterministic recommenders. If additional statistical comparisons are planned, add them before reading test outcomes. Holm correction is applied within the named family, including all its comparisons; failed/missing comparisons are not silently removed. Optional HRL-PRP/ROTAN are outside the required matrix.

Main method and learned internal policy runs use 11/29/47. Deterministic popularity references have one result. Supplementary controllers use 11; timing repeats remain the same trained checkpoint. State-removal/source-removal and every budget/lambda/cap sensitivity variant retrain the controller with the predictor unchanged. M and backbone changes fit new predictors. All-context may exceed both budget and acquisition cap; M=1 disagreement is zero.

## Complexity interpretation

For L recent events, R records per block, dimension d, J sources, M members, catalog N and A acquisitions, the Transformer base costs O(M(L²d + Ld²)); GRU base O(MLd²). Historical source GRUs cost O(MARd²), plus attention pooling. Fusion refresh and full scoring recur A+1 times, with scoring O(M(A+1)Nd); Q decisions cost O((A+1)(state·q + q² + qJ)). The base and each acquired source encoding run once per request. All runtime measurements include descriptors and source retrieval.

The training subset lattice stores O(requests·2^J·d) latents and statistics. It does not retain full-catalog probabilities and is never used to claim serving latency. Its exponential dependence on J should be discussed. Full-softmax predictor training still holds batch-by-catalog activations; output chunking does not eliminate that total autograd storage.

`complexity` varies active scoring catalog size, recent-history cap, fixed-source acquisition target and active ensemble members one factor at a time, using trained weights and real request prefixes. It measures raw workload scaling under a controlled fixed policy. It does not evaluate recommendation quality under truncated catalogs or claim that a policy trained for one workload remains optimal for another. Source unavailability can reduce actual acquisitions, which is reported. All loaded ensemble weights stay resident during this sweep; report separate M=1/M=3 deployment memory from their dedicated latency jobs. CPU process RSS and CUDA peak tensor allocation measure different quantities; neither is total GPU driver memory.

Raw timing assumes warm model weights and an initialized historical index. Startup, conversion, graph/index construction and network downloads are excluded. It measures serial batch-one inference, not concurrent serving or a wall-clock budget guarantee. The additive profiled reward costs are separate from measured deployment latency.
