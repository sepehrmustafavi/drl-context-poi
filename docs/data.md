# Dataset sources and exact preparation

## Foursquare NYC / Tokyo

Obtain the city check-in files from the authors' dataset page:
https://sites.google.com/site/yangdingqi/home/foursquare-dataset

The supported raw layout is tab-separated, with no header:

`user ID, venue ID, venue category ID, venue category name, latitude, longitude, timezone offset, UTC timestamp`

Use `convert --format foursquare`. The converter parses the timestamp in UTC, preserves observed coordinates/category IDs, removes exact duplicate observations and writes a SHA-256 manifest. It does not download or infer social links. UTC is also the current feature convention; the timezone-offset field is not used for local civil-time/DST claims.

## Gowalla

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
