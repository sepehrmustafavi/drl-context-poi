# Dataset sources and exact preparation

## Foursquare NYC / Tokyo

Obtain the city check-in files from the authors' dataset page:
https://sites.google.com/site/yangdingqi/home/foursquare-dataset

The supported raw layout is tab-separated, with no header:

`user ID, venue ID, venue category ID, venue category name, latitude, longitude, timezone offset, UTC timestamp`

Use `convert --format foursquare`. The converter parses the timestamp in UTC, preserves observed coordinates/category IDs, removes exact duplicate observations and writes a SHA-256 manifest. It does not download or infer social links. UTC is also the current feature convention; the timezone-offset field is not used for local civil-time/DST claims.

## Pinned Gowalla-CA release

The final supported release is `dataset/Gowalla-CA.zip` from GETNext commit `6a79081aec6951ebec31bff6400d6ba0e40a613e`:

[Exact source archive](https://github.com/songyangco/GETNext/blob/6a79081aec6951ebec31bff6400d6ba0e40a613e/dataset/Gowalla-CA.zip)

SHA-256: `c5c0604ab2ee68219338541bc23fe7e9f2023c6e9569bafda68488bde87afc3f`.

Use `convert --format getnext-gowalla`. It rejects a different archive hash, reads only the three named check-in CSV members, merges their observations, discards precomputed graph files and original split assignments, removes exact duplicate canonical observations and performs the new chronological split through `Store`. Categories in the archive have undocumented availability timing, so the adapter conservatively disables them. Use `configs/gowalla.yaml`.

Name this dataset **Gowalla-CA GETNext release**, identifying the commit and inherited upstream preprocessing. It is a released, previously filtered subset; neither complete unfiltered California SNAP data nor an exact polygon extraction is claimed. There is no new future-based user/POI filtering in our converter. Its measured metadata and partition audit are in `gowalla-ca-audit.json`; these are data statistics, not trained-model evaluation results. The canonical checksum refers to the tested conversion runtime and should be regenerated/recorded for your own data preparation.

## Alternative SNAP geographical variant

Official SNAP source:
https://snap.stanford.edu/data/loc-gowalla.html

```bash
python scripts/download.py --url https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz --out data/raw/loc-gowalla_totalCheckins.txt.gz
```

SNAP rows are `user, UTC time, latitude, longitude, location ID`. The converter inserts a constant missing-category token. Configure sources as `[remote, periodic, transition]`, order `[0,1,2]`, and cap 3; no real categories or social graph are assumed.

The README's geographical rectangle is south=32.5, north=42.0, west=-124.5, east=-114.1, inclusive. This is an explicitly reproducible **California-region bounding-box** extraction, including some neighboring-state locations. It is not the exact California polygon and not asserted to match any paper's Gowalla-CA release. In the article, use an accurate variant name and publish the raw checksum, bounding box, canonical checksum, event counts, users, POIs and split cutoffs. If exact California boundaries are required, prepare a polygon-filtered canonical CSV externally and record the polygon/version/checksum rather than calling the rectangle an exact state boundary.

## Canonical CSV

Required header:

```csv
user_id,poi_id,timestamp,latitude,longitude,category
u1,p1,2013-01-01T10:00:00+00:00,40.7,-74.0,food
```

Identifiers are strings, timestamps must be parseable in UTC, and coordinates must be finite and in range. Every row is one observed event. Convert canonical input with `--format canonical` to normalize ordering and produce a manifest. Do not reuse an upstream per-user random split.

The output catalog uses fitting events only. New targets stay out of catalog; new historical inputs use UNK. Filtering is limited to exact duplicate observations and the declared geography. Minimum history applies at request creation, using only previous events, not future user activity. The implementation does not silently remove cold-start users or out-of-catalog test targets.

## Required publication audit

Run `audit` for every final dataset and attach its JSON to the release. Document time range, four cutoffs, fitting-only vocabulary, minimum history, geographical definition, repeat handling and all/in-catalog request counts. Report `canonical_sha256` in every exported baseline manifest. Do not distribute raw data unless permitted by the original provider's terms.
