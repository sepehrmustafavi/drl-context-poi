# Chapter 4 evaluation mapping

| Section | Required evidence | Implementation / output |
|---|---|---|
| 4.1 Hardware and Software Environment | GPU, Python/PyTorch/CUDA, seed, commit, thread count | Predictor `environment.json`, `config.json`, profiling output; environment script |
| 4.2 Datasets | NYC, Tokyo, declared Gowalla geography; time separation, catalog, OOC targets | `convert`, manifests, `audit`; `docs/data.md` |
| 4.3 Evaluation metrics | Recall@5/10/20, NDCG@10, full-rank MRR, NLL, cost, acquisition count; request/user macro | `metrics.py`, per-request JSONL, summary JSON |
| 4.4 Hyperparameter Optimization | Bounded validation-only selection; one fixed seed; final three seeds | `hpo`, frozen per-dataset YAMLs; controller validation sweep as separate declared jobs |
| 4.5 Overall Performance | Final proposed method vs external recommenders on three datasets, three seeds | Internal jobs ready; external official adapters still required |
| 4.6 Ablation Study | Base/all, forced acquisition, no uncertainty/representation, source removal, M=1 | Plan: main references three seeds; extra ablations fixed seed 11; source/M studies NYC/Tokyo |
| 4.7 Complexity Analysis | Parameters, serial raw p50/p95 latency, peak allocated memory, asymptotic discussion | `latency`; run three timing repetitions on idle reference hardware, outside shared-worker plan |
| 4.8 Baseline Comparison | DDQN/DQN vs direct selector, bandit over all feasible subsets, myopic, fixed, entropy, random | Main learned comparisons all datasets/seeds; same predictor/cache/cost vector |
| 4.9 Discussions | Budget/cost/cap sensitivity, short/long histories, case traces, failures, limitations | NYC/Tokyo seed-11 variant jobs; `analysis.subgroups`; per-request `trace` fields |

## Main suite

- Three datasets × three seeds × independently trained three-member Transformer ensemble.
- DDQN, DQN, learned myopic, direct full-feedback subset selector and subset contextual bandit.
- Base-only and unconstrained all-context; validation-selected fixed subset; fixed-order entropy and random-valid policies.
- Global popularity and personal frequency (deterministic references; no artificial three-seed variance).
- Three-seed SASRec-style internal baseline. It is not a verified official SASRec reproduction.
- GETNext/DCHL/iPCM planned through audited official adapters, which remain an explicit implementation gap.

## Single-seed supplementary suite

- No adaptive STOP; zero uncertainty state; zero latent representation state, all three datasets.
- For NYC/Tokyo: remove each source and retrain the controller; independent M=1 predictor and controller; matched contextual GRU.
- NYC/Tokyo budget fractions 0, .25, .5, .75, 1; lambda 0, .03, .1, .3, 1; acquisition cap 1..J. The baseline setting supplies .5/.1/J, so the plan avoids duplicating it.
- All these are **full-data final experiments**, not pilots. Single-seed sensitivity supports conditional descriptive trends, not strong significance claims across random initializations.

The generator preserves fixed predictor identities for policy ablations and retrains each policy. Changing budget, lambda, cap or disabled sources at inference alone is rejected by the checkpoint loader. M/backbone changes require new predictor checkpoints/caches.

## Analyses after main runs

Use `drlpoi.analysis.subgroups` with bins fitted to fitting histories. Inspect both successful and unsuccessful traces; do not select only favorable anecdotes. Compare quality/cost using a validation-declared absolute NDCG tolerance (default .002); do not choose the tolerance after seeing test results.

`compare` gives a paired user-cluster confidence interval and conditional sign-flip p-value for one trained pair. Apply `analysis.holm` to the predeclared comparison family. Show three-seed mean/std separately. Analysis at one seed must be labeled explicitly.

## Complexity statement

For sequence length L, source bound R, dimension d, J sources, M members, catalog N and K acquisitions: Transformer attention is O(M L² d) plus projection/feed-forward terms; GRU is O(M L d²). Source pooling/encoding is bounded by selected blocks (history GRUs include O(M R d²)). Full-catalog output projections cost O(M N d) per refresh. A sequential episode can require K+1 refreshes, plus policy inference and retrieval. Policy state/Q computation is small, but repeated full-catalog refreshes need measurement.

The subset training cache has at most 2^J states per request and stores O(2^J d) latents plus statistics/rewards/ranks; it does not store all catalog probabilities. It is practical for J=3/4, and scales exponentially in J. It is a training acceleration, not a serving speed claim. Full-softmax training retains B×N logits; reducing output chunk size does not eliminate total autograd memory.

The historical index still costs memory and descriptors/retrieval cost CPU time. The raw latency path measures those operations but excludes process startup, model loading, data conversion and network access. GPU peak allocation is not total process/RSS memory; report that distinction.
