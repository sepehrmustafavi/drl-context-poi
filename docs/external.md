# Pinned upstream models under the common protocol

`src/drlpoi/upstream.py` loads verified model classes from the commits in `configs/upstream.json`. It trains/evaluates them directly with common request IDs, fit-only catalog, global 50/20/10/20 time split, full catalog scoring, validation checkpoint selection and every test request. Stock upstream entry points are not executed: their session splits, preprocessing and test-driven checkpoint logic do not satisfy this experiment.

These are **common-protocol adaptations**, not numerical reproductions of the original published tables. Keep that qualification in the article. No upstream source is vendored or relicensed. Fetch source and read its license/terms before distribution.

```bash
python -m pip install -e '.[baselines]'
python scripts/fetch_upstream.py getnext
python -m drlpoi.cli baseline --csv data/canonical/nyc.csv --model getnext --out runs/nyc/getnext/seed11 --resume
```

The full plan schedules each of SASRec, GETNext, DCHL and iPCM on each dataset and seed. `baseline-latency` measures their raw inference separately. Training state includes optimizer/RNG state and resumes at epoch boundaries. Provenance includes commit, model-file SHA-256, adapter configuration, data identity and environment.

## Shared adaptations

- Fit labels come only from the predictor-fit interval. Controller-fit labels train acquisition policies and are not added to external recommender training.
- Fit graphs and candidate metadata are built from the complete fit interval and frozen. This follows the external models' train-graph convention. It does **not** make the stronger claim that each fitting example sees only earlier fit-graph edges. The proposed contextual predictor separately uses request-time edge cutoffs. No policy/validation/test event is used to fit an external graph, cluster or candidate descriptor.
- The temporal range is globally partitioned. A request supplies a strictly earlier observed prefix; its next timestamp/location/category is never an input. For GETNext/iPCM, target time/category/region may supervise auxiliary losses **during fit only**, as in the original models.
- The configured recent window bounds sequential input. This changes the upstream trajectory/session preprocessing and is disclosed. Every test request remains evaluated, including unknown users and out-of-catalog targets; previously visited items remain rankable.
- Metadata for catalog nodes uses their first fitting observation. Unknown historical POIs use a zero input vector or mean known node/region embedding. Unknown users use mean fitted user representations/incidence, rather than being silently removed. No future identity receives a trained user embedding.
- Shared configuration values control dimension, depth, dropout, batch size, LR and epoch/patience limits. Model-specific graph settings are explicit in the adapter (`geo_radius_km`, default 2.5 km; `region_clusters`, default 100, clipped to catalog size). Record the actual settings in comparisons. The predictor/controller HPO does not claim to optimize each baseline's own best configuration.

## SASRec

The pinned maintained PyTorch implementation supplies item/position embeddings, self-attention and feed-forward blocks. An input-only zero UNK row is added; full ranking scores only the real training catalog. The original sampled-positive/negative BCE objective is used at the final position of each common next-event request. Negative sampling excludes fitting visits; if a user's fitting visits cover the entire catalog, it samples among items other than the positive target so the sampler cannot hang. No negative is needed for a one-item catalog. Sequences are right-aligned as in the upstream interface (leading zeros, latest observation at the last position). The pinned implementation uses its `norm_first=True` option. This is a maintained implementation adaptation, not a claim of matching the 2018 paper's exact TensorFlow training schedule.

## GETNext

Uses upstream GCN, node flow attention, user/category/Time2Vec embeddings, fusion layers and multi-head Transformer decoders. The flow graph contains fitting consecutive transitions; self loops and random-walk normalization match the upstream GCN path. Flow attention rows are computed only for requested last POIs using the upstream algebra; a dense N-by-N attention matrix is unnecessary. The objective combines POI and category cross-entropy and 10× next-time MSE, with target values used only in fitting losses. Individual sequences avoid artificial padding influencing the official Transformer. UTC day fraction is the declared time feature. Missing categories use a constant missing token and carry no real category signal.

## DCHL (SIGIR 2024)

Uses the upstream model's multi-view, directed and geographic graph convolutions, gates, fusion and contrastive objectives. User–POI incidence is binary and row-normalized in both directions. Directed edges include all ordered earlier/later POI pairs within each user's fit trajectory, following the upstream graph function; it is not replaced by only adjacent edges. A BallTree constructs the 2.5-km geographic graph without a dense all-pairs distance matrix.

The official forward currently scores a user from structural graph representations and does not consume the supplied recent prefix. The adapter preserves this behavior rather than silently adding a sequence encoder. An unknown user uses mean fitted incidence. Two numerical/runtime fixes are explicit: functional dropout is disabled in evaluation; InfoNCE is evaluated with stable logsumexp and checkpointed chunks, with negligible numerical differences from the original epsilon formula. Unused contrastive loss is skipped at inference. POI contrastive computation remains quadratic in catalog size; its time cost is not claimed to disappear.

## iPCM (AAAI 2025) compatibility

The source uses observed trajectory time as input and **predicted** next-time/region in its ranking adjustment. It therefore supports this problem without future target time/location at inference. The adapter retains upstream gated POI/region graphs, user mean-POI personalization, time embedding, fusion, Transformer heads, auxiliary losses and ranking adjustment.

Spatial KMeans uses first-fit node coordinates and a fixed seed. Contiguous temporal clustering follows the upstream 72-bin, 20-group dynamic program on fit-only category distributions, adding explicit zero handling for empty bins. The same learned bin map transforms later observed times. Gowalla has no trusted category metadata, so its missing-category bin supplies no semantic category information; the resulting temporal clustering can be uninformative and must be discussed. No category acquisition source is enabled there.

The original adjustment adds a binary personal-visit indicator to softmax POI scores for the top predicted regions/times. The adapter clips top-k counts to the available number of classes (so small fixtures/catalogs work) and **normalizes the adjusted positive scores** for NLL. This preserves adjusted ranking but defines an explicit normalized-score likelihood; do not call it a calibrated generative probability. Test outcomes never select the checkpoint.

## Independent external exchange

`export-external` and `import-external` remain available for another audited implementation. Exported `*_features.jsonl` files contain prefix indices and query time without targets; `*_requests.jsonl` contain labels for loss/evaluation only. Rebuild all maps and graphs from `predictor_fit_events.csv`; later events are accessible only through each request's prefix.

Imported JSONL must contain every request exactly once and a finite full-catalog vector in exported order. Manifest checks cover data/catalog/request hashes, seed, source commit, train split and explicit temporal-audit declarations. The importer interprets scores as logits. Export log probabilities if importing normalized adjusted iPCM probabilities; do not pass those probabilities as logits and then interpret the resulting softmax as the same NLL. The native adapters already calculate their intended NLL directly.
