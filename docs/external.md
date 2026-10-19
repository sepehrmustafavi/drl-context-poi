# Official external baselines: integration contract and current gap

GETNext (SIGIR 2022), DCHL (SIGIR 2024) and iPCM (AAAI 2025) are relevant external comparators. Their pinned public repository commits are in `configs/upstream.json`. Official code is fetched into ignored working storage with `scripts/fetch_upstream.py`. It is not vendored, relicensed or silently replaced by a small local graph model.

**Current status: upstream common-protocol adapters are not implemented.** Fetching source or exporting data is not a reproduction. The main internal pipeline runs independently of these adapters, but the planned strong external comparison is incomplete until they are implemented, tested and executed on the real datasets.

## Why stock training scripts cannot be blindly mixed

The reviewed repositories have their own dataset/session formats, preprocessing, graph constructions, user/item maps and inference assumptions. The common experiment instead requires globally chronological requests, a fitting-only catalog, complete coverage of cold users, no future target timestamp and common full-catalog ranking. An adapter must resolve those differences explicitly. GETNext also has an older documented dependency environment. Use separate environments rather than forcing all original dependencies into this package.

## Export

```bash
python -m drlpoi.cli export-external --csv data/canonical/nyc.csv --out artifacts/exchange/nyc
```

Files:

- `catalog.json`: exact ordered POI IDs. Model output score index 0 corresponds to this catalog's first entry.
- `predictor_fit_events.csv`: allowed parameter-fitting observations.
- `events.csv`: all chronological observations, available for constructing a request's past prefix only. It contains later rows, so feature code must obey each request's prefix index list.
- `{split}_features.jsonl`: request ID, user ID, earlier prefix indices, query timestamp; no label.
- `{split}_requests.jsonl`: the same data plus target index, for loss/evaluation only.
- `manifest.json`: catalog, data and request-set fingerprints and audit.

Fitting labels must come only from `fit`. Later policy-fit labels are reserved for training the proposed controller and should not be silently added to an external predictor. If a second equal-total-label-budget comparison is desired, declare and rerun **every** predictor with the same additional data; it is a distinct experiment.

## Required adapter behavior

1. Rebuild mappings, graphs and any clustering from allowed fitting observations. For graph-dependent features during fitting, use historical snapshots if making the same strict per-request temporal claim; otherwise explicitly document the weaker train-only transductive graph convention and run a matched control.
2. Use the observed query time, not the held-out target timestamp. Preserve a separate target-time-aware variant only as a clearly different task.
3. Resolve unseen users using a declared, trained fallback; do not omit them. Map unknown historical POIs separately from the output catalog. Never allocate an unseen target as an evaluated output class.
4. Select epochs/hyperparameters on the common validation set; never reuse the upstream test split or test-driven settings.
5. Emit a finite score for every fitting-catalog POI and every common test request, including out-of-catalog targets (which score as misses).
6. Record the source commit, adapter commit, source license, exact configuration, software environment and seed. Inspect whether the method's original training objective materially changed and label adaptations honestly.

## Output exchange

One JSON object per request in a `.jsonl` file:

```json
{"request_id":"r123", "scores":[0.5, -0.2, 1.7]}
```

The score list must contain one score per POI in the exact exported catalog order, not only top-20 candidates. Scores are logits or unnormalized finite ranking scores. The importer computes a softmax NLL; do not interpret that NLL as calibrated likelihood unless the original scores support that interpretation.

Create provenance JSON with values copied from the export manifest:

```json
{
  "model_name":"DCHL-common-protocol",
  "source_commit":"actual-upstream-and-adapter-commit",
  "seed":11,
  "training_config":{"actual":"configuration"},
  "data_identity_sha256":"copy-data_identity_sha256",
  "catalog_sha256":"copy-catalog_sha256",
  "request_ids_sha256":"copy-request_ids_sha256.test",
  "train_split":"fit",
  "target_time_used":false,
  "temporal_leakage_audited":true
}
```

Those strings illustrate schema fields, not valid benchmark provenance. Supply the actual hashes/commit after auditing the adapter.

```bash
python -m drlpoi.cli import-external --csv data/canonical/nyc.csv --scores artifacts/dchl_scores.jsonl --manifest artifacts/dchl_provenance.json --out runs/nyc/dchl/seed11/test
```

The importer refuses missing/duplicate/unknown requests, different catalog/data fingerprints and invalid scores. Its audit declaration is a required attestation, not an automated proof of causal or temporal validity.
