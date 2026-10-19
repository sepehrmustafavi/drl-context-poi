# Reproducibility and methodological protocol

## Scope

The finite-horizon RL episode is one recommendation request. Actions acquire at most one bounded block from a previously unselected source or STOP. Acquiring information changes the agent's observation, not the observed user's trajectory. Contextual bandits choosing entire subsets are strong and necessary comparators. Better sequential performance is an empirical question, not guaranteed by using Double DQN.

## Temporal separation

The global timestamp range is split 50% predictor fitting, 20% controller fitting, 10% validation and 20% test by elapsed time. These are time-range fractions, not exact proportions of events. Requests are assigned using target timestamps, while the model receives only earlier prefix events. Query time is the last observed timestamp. No target time, coordinates or category enter features. The known output catalog and category vocabulary use the predictor-fit interval only. Unknown historical POIs have an UNK input embedding.

Collective edges used in predictor fitting are strictly earlier than the request time. Later requests use a frozen predictor-fit graph, with cutoff `min(query_time, predictor_fit_boundary)`; a prefix crossing a boundary therefore cannot read a later graph. Recent/remote/periodic user histories continue growing with actual past observations, without changing predictor weights. Source metadata and observed prefix histories may update; population graph parameters do not.

There is no future-frequency filtering or per-user random split. Exact duplicate rows are removed during conversion. Repeated visits remain valid. The full fitting catalog is ranked, including previously visited POIs. Unknown test targets count as misses in all-request metrics and are excluded from in-catalog NLL. Both request-macro and equal-user-macro metrics are reported.

## Predictor

Each independent ensemble member owns a POI/category/time/space encoder, a causal Transformer (or GRU), source encoders, attention fusion and tied output embeddings plus bias. Fit members independently with empty, full and random subsets. Select each member using validation NDCG averaged across empty/full conditions. This is explicitly the implemented predictor-selection criterion; the controller subsequently has its own validation selection.

The sequence-only baseline is **SASRec-style**: causal Transformer, POI and position embeddings, full-softmax loss. It is a local controlled baseline, not an exact reproduction of every original SASRec training detail. Do not label it a verified official SASRec reproduction. Exact official SASRec and the graph baselines can use the external exchange contract.

Embeddings use standard deviation .02 with zero padding/UNK initialization; linear/attention projections use Xavier initialization, including independently initialized Transformer layers. A single ensemble member is trained at a time. Policy training freezes the entire ensemble. Mean predictive probabilities determine NLL and ranks. Entropy and Jensen–Shannon disagreement are descriptive uncertainty features, not calibrated confidence estimates. M=1 has zero disagreement.

## Context and state

Remote history excludes the recent prefix. Periodic history uses prior weekday/hour matches, using UTC consistently. Historical blocks deduplicate by POI, taking recent occurrences. Transition and category-conditioned blocks select the most frequent earlier successor POIs and a historical representative event. Blocks are bounded by the configured record cap; overlapping POIs across different sources remain allowed. No future POI metadata is used to build a historical record.

Gowalla has no genuine category field. Its final configuration uses remote/periodic/transition sources only; missing categories are a constant input token, not inferred labels. No social source is fabricated.

State is mean latent representation, selected and available masks, log-count descriptors, cost vector, normalized predictive entropy/disagreement, remaining budget fraction and acquisition fraction. Initially unacquired block embeddings are absent. This compressed observation need not be an exact Markov state of human preferences.

## Return and constraints

Acquisition reward is `-lambda_cost * c_j`; STOP reward is `-NLL(target | selected context)`. Discount is exactly 1. Every episode has an explicit STOP transition, including when no acquisition remains possible. Source selection is masked by availability, prior selection, remaining budget and the acquisition cap. With budget zero the only valid action is STOP.

A fixed reference cost vector is profiled from fitting requests at batch size one. Costs include retrieval, feature construction, source encoding and ensemble prediction refresh. Each source's median is normalized by the pooled median. Unobserved sources receive the pooled median as a documented fallback and remain action-masked where unavailable. The vector is shared across compared policies and seeds of a dataset/backbone. Do not reprofile it separately on each evaluated GPU and then claim identical budgets.

Cached validation runs batch controller decisions to avoid thousands of single-request GPU calls. Raw serving remains batch size one. Numerical agreement between cached and raw inference must be checked on the final CUDA software/hardware as well as the CPU tests. The additive proxy is not wall-clock latency. Raw latency includes cold per-request retrieval/block caching and repeated prediction refreshes, with warm model weights and a warm historical index. This is neither disk-cold startup latency nor a concurrent-serving benchmark.

## Policies

- Double DQN: replay, masked online argmax and target-network evaluation, Huber loss, explicit terminal handling.
- DQN: target-network maximum with the same architecture, reward and resource constraints.
- Myopic: supervised one-step NLL improvement minus acquisition cost, STOP value zero; no multi-step bootstrapping.
- Direct selector: full-feedback supervised cross-entropy over all feasible subsets, target maximizes training NLL/cost objective.
- Contextual bandit: one selected feasible subset per training interaction; regression uses only that subset's observed reward.
- Best fixed subset: global subset chosen on validation NDCG, then lower cost. Intersect with availability; abstain if the resulting subset exceeds the request budget/cap.
- Entropy: fixed source order and a validation-selected normalized entropy threshold.
- Random: uniformly selects among currently valid source actions and STOP, with a fixed RNG seed.
- Oracle: target-dependent diagnostic minimizing terminal NLL plus additive cost among feasible subsets. It is **not** a guaranteed upper bound on NDCG.
- All-context: uses every available source and is explicitly unconstrained; it may exceed the policy budget or cap.

Policy checkpoint selection uses validation NDCG@10, with lower mean cost for exact ties. Predictor and controller checkpoint selection share validation, so HPO must be bounded and declared; test labels are not selection criteria.

## Experiments and HPO

Main seeds are 11, 29, 47 on NYC, Tokyo and the documented Gowalla variant. Main learned policies are DDQN, DQN, myopic, direct and bandit. Supplementary architecture/sensitivity runs use fixed seed 11, never a favorable seed selected after seeing outcomes. The GRU and M=1 studies use NYC/Tokyo only.

`hpo` evaluates at most six full-data predictor configurations (dim 64/128; LR .0003/.001/.003) using seed 11. It preserves the requested number of ensemble members; changing M to save HPO compute is a protocol change that must be declared. These are predictor-only trials. For controller HPO, train candidates on policy-fit, choose on validation, and freeze lambda/budget/threshold before generating the final plan. Sensitivity runs then vary one factor around that declared choice and retrain the controller. No test-driven selection is permitted. Shared predictor fits may be reused where identities match.

Leave-one-source-out removes that source's action, availability and count descriptor, and retrains the policy over the same frozen predictor. It does not retrain a new predictor without that source. The no-representation ablation zeroes the entire latent state vector while retaining the other state components. The no-uncertainty ablation zeroes both entropy and disagreement and retrains.

## Statistics and limitations

Show mean and sample standard deviation over the three main seeds. User-cluster bootstrap quantifies paired request-sample variation conditional on a trained seed. Cluster sign-flip testing assumes exchangeability of paired user-level differences under the null. Predeclare comparison families and correct p-values with Holm. Report seed variability separately; do not pool three seeds as independent users.

Offline check-ins support evaluating next-recorded-POI accuracy and information acquisition utility. They do not establish that recommendations change real user behavior, improve long-term user utility or have causal online effects. A stronger or simpler baseline may win. Document that result without changing the test protocol.
