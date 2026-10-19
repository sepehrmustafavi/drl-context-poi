"""Chronological requests and historical-only, bounded context retrieval."""

from __future__ import annotations
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from collections.abc import Sequence
from itertools import islice
from pathlib import Path
import gzip
import math
import numpy as np
import pandas as pd
from .util import file_hash, write_json

REQUIRED = ["user_id", "poi_id", "timestamp", "latitude", "longitude", "category"]
SOURCES = ("remote", "periodic", "transition", "category")


def convert(raw, output, fmt, bbox=None):
    if fmt == "canonical":
        df = pd.read_csv(raw, dtype={"user_id": str, "poi_id": str, "category": str})
    elif fmt == "foursquare":
        names = [
            "user_id",
            "poi_id",
            "category",
            "category_name",
            "latitude",
            "longitude",
            "utc_offset",
            "timestamp",
        ]
        df = pd.read_csv(
            raw,
            sep="\t",
            names=names,
            encoding="latin-1",
            dtype={"user_id": str, "poi_id": str, "category": str},
        )
    elif fmt == "gowalla":
        opener = gzip.open if str(raw).endswith(".gz") else open
        with opener(raw, "rt") as f:
            df = pd.read_csv(
                f,
                sep="\t",
                names=["user_id", "timestamp", "latitude", "longitude", "poi_id"],
                dtype={"user_id": str, "poi_id": str},
            )
        df["category"] = "__missing__"
    else:
        raise ValueError(fmt)
    for col in REQUIRED:
        if col not in df:
            raise ValueError(f"Missing column: {col}")
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, format="mixed", errors="raise")
    for col in ("latitude", "longitude"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if not df.latitude.between(-90, 90).all() or not df.longitude.between(-180, 180).all():
        raise ValueError("Coordinates out of range")
    if df[["user_id", "poi_id", "timestamp", "latitude", "longitude"]].isna().any().any():
        raise ValueError("Null required fields")
    df["category"] = df.category.fillna("__missing__").astype(str)
    if bbox:
        south, north, west, east = bbox
        df = df[df.latitude.between(south, north) & df.longitude.between(west, east)]
    # Exact duplicate observations only; repeated visits at different times are retained.
    df = df[REQUIRED].drop_duplicates().sort_values(["timestamp", "user_id", "poi_id"], kind="stable")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    write_json(
        str(output) + ".manifest.json",
        dict(
            raw_sha256=file_hash(raw),
            canonical_sha256=file_hash(output),
            format=fmt,
            bbox=bbox,
            events=len(df),
            users=df.user_id.nunique(),
            pois=df.poi_id.nunique(),
            time_convention="UTC; request time = last observed event; no future target timestamp input",
        ),
    )


def haversine(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(min(1, math.sqrt(max(0, h))))


class Prefix(Sequence):
    """Immutable-length view: sharing user histories avoids quadratic prefix storage."""

    def __init__(self, indices, end):
        self._indices = indices
        self._end = end

    def __len__(self):
        return self._end

    def __iter__(self):
        return islice(self._indices, self._end)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._indices[i] for i in range(*key.indices(self._end))]
        key = key + self._end if key < 0 else key
        if not 0 <= key < self._end:
            raise IndexError(key)
        return self._indices[key]


@dataclass(frozen=True)
class Request:
    id: str
    user: str
    prefix: Prefix
    target: int
    split: str


class Store:
    def __init__(self, csv, cfg):
        self.path, self.cfg = str(csv), cfg
        if min(cfg["min_history"], cfg["recent"], cfg["block_size"]) < 1 or not cfg["sources"]:
            raise ValueError("Positive history/block bounds and at least one source are required")
        self.df = pd.read_csv(csv, dtype={"user_id": str, "poi_id": str, "category": str})
        if not set(REQUIRED) <= set(self.df):
            raise ValueError("Convert the data to the canonical schema first")
        if self.df[REQUIRED].isna().any().any():
            raise ValueError("Canonical inputs contain null fields; normalize with convert first")
        if not self.df.latitude.between(-90, 90).all() or not self.df.longitude.between(-180, 180).all():
            raise ValueError("Invalid coordinates")
        self.df = self.df.sort_values(["timestamp", "user_id", "poi_id"], kind="stable").reset_index(
            drop=True
        )
        self.t = pd.to_datetime(self.df.timestamp, utc=True, format="mixed").astype("int64").to_numpy() / 1e9
        if not len(self.t):
            raise ValueError("Empty dataset")
        f = cfg["split_fractions"]
        if len(f) != 4 or any(x <= 0 for x in f) or not np.isclose(sum(f), 1):
            raise ValueError("Four positive split fractions must sum to one")
        # Boundaries interpolate elapsed time, not per-user event quantiles.
        self.cuts = self.t[0] + np.cumsum(f)[:-1] * (self.t[-1] - self.t[0])
        fit = self.df[self.t < self.cuts[0]]
        self.catalog = sorted(fit.poi_id.unique())
        if not self.catalog:
            raise ValueError("No predictor-fit catalog")
        self.pmap = {p: i + 1 for i, p in enumerate(self.catalog)}  # 0 PAD; N+1 historical UNK
        self.unk = len(self.catalog) + 1
        self.cmap = {p: i + 1 for i, p in enumerate(sorted(fit.category.unique()))}
        self.p = np.array([self.pmap.get(x, self.unk) for x in self.df.poi_id])
        self.cat = np.array([self.cmap.get(x, 0) for x in self.df.category.fillna("__missing__")])
        self.coords = self.df[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        self.coord_mean = fit[["latitude", "longitude"]].mean().to_numpy()
        self.coord_std = fit[["latitude", "longitude"]].std(ddof=0).to_numpy().clip(1e-4)
        self.requests = []
        self.edges, self.cedges = defaultdict(list), defaultdict(list)
        users = defaultdict(list)
        for idx, row in enumerate(self.df.itertuples()):
            h = users[row.user_id]
            if len(h) >= cfg["min_history"]:
                split = ("fit", "policy", "val", "test")[bisect_left(self.cuts, self.t[idx] + 1e-6)]
                self.requests.append(Request(f"r{idx}", row.user_id, Prefix(h, len(h)), idx, split))
            if h and self.t[idx] < self.cuts[0]:
                self.edges[int(self.p[h[-1]])].append((self.t[idx], idx))
                self.cedges[int(self.cat[h[-1]])].append((self.t[idx], idx))
            h.append(idx)
        self.sources = list(cfg["sources"])
        if len(set(self.sources)) != len(self.sources) or not set(self.sources) <= set(SOURCES):
            raise ValueError("Supported sources: " + str(SOURCES))
        self.identity = dict(
            csv_sha256=file_hash(csv),
            cuts=self.cuts.tolist(),
            catalog=self.catalog,
            min_history=cfg["min_history"],
            recent=cfg["recent"],
            block_size=cfg["block_size"],
            sources=self.sources,
            periodic_hour_window=cfg["periodic_hour_window"],
        )
        self._collective_cache = {}
        self._times = {
            kind: {k: [t for t, _ in v] for k, v in mapping.items()}
            for kind, mapping in [("transition", self.edges), ("category", self.cedges)]
        }

    def split(self, name, eligible=False):
        return [
            r for r in self.requests if r.split == name and (not eligible or self.p[r.target] != self.unk)
        ]

    def target(self, r):
        return int(self.p[r.target] - 1) if self.p[r.target] != self.unk else -1

    def _history(self, r, source):
        old = r.prefix[: -self.cfg["recent"]] if len(r.prefix) > self.cfg["recent"] else ()
        if source == "remote":
            return list(old)
        tau = self.t[r.prefix[-1]]
        day = int(tau // 86400) % 7
        hour = tau / 3600 % 24
        w = self.cfg["periodic_hour_window"]
        return [
            i
            for i in old
            if int(self.t[i] // 86400) % 7 == day
            and min(abs(self.t[i] / 3600 % 24 - hour), 24 - abs(self.t[i] / 3600 % 24 - hour)) <= w
        ]

    def _edge_prefix(self, r, source):
        key = int(self.p[r.prefix[-1]]) if source == "transition" else int(self.cat[r.prefix[-1]])
        if source == "category" and key == 0:
            return [], 0
        mapping = self.edges if source == "transition" else self.cedges
        # Strictly earlier edges prevent same-timestamp label contamination.
        cutoff = min(self.t[r.prefix[-1]], self.cuts[0])
        n = bisect_left(self._times[source].get(key, []), cutoff)
        return mapping.get(key, []), n

    def descriptors(self, r):
        counts = []
        for s in self.sources:
            n = len(self._history(r, s)) if s in ("remote", "periodic") else self._edge_prefix(r, s)[1]
            counts.append(n)
        return np.array(counts) > 0, np.log1p(np.array(counts, dtype=np.float32))

    def retrieve(self, r, j):
        source = self.sources[j]
        cap = self.cfg["block_size"]
        if source in ("remote", "periodic"):
            seq = self._history(r, source)
            # One observation per POI within each block; latest observation wins.
            seen, out = set(), []
            for i in reversed(seq):
                if self.df.poi_id.iat[i] not in seen:
                    seen.add(self.df.poi_id.iat[i])
                    out.append(i)
                if len(out) == cap:
                    break
            return list(reversed(out))
        _, n = self._edge_prefix(r, source)
        # Repeated validation/test requests reuse a frozen indexed neighbor block.
        key = int(self.p[r.prefix[-1]]) if source == "transition" else int(self.cat[r.prefix[-1]])
        return list(self._collective(source, key, n, cap))

    def _collective(self, source, key, n, cap):
        cache_key = (source, key, n, cap)
        if cache_key in self._collective_cache:
            return self._collective_cache[cache_key]
        mapping = self.edges if source == "transition" else self.cedges
        counts, latest = Counter(), {}
        for _, i in mapping.get(key, [])[:n]:
            counts[int(self.p[i])] += 1
            latest[int(self.p[i])] = i
        result = tuple(latest[p] for p, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:cap])
        if len(self._collective_cache) >= 8192:
            self._collective_cache.clear()
        self._collective_cache[cache_key] = result
        return result

    def features(self, indices):
        # [hour sin/cos, weekday sin/cos, log elapsed, start, lat/lon, log distance]
        out = []
        for k, i in enumerate(indices):
            t = self.t[i]
            dt = max(0, t - self.t[indices[k - 1]]) if k else 0
            dist = haversine(self.coords[i], self.coords[indices[k - 1]]) if k else 0
            a, b = t / 86400 * 2 * np.pi, t / (86400 * 7) * 2 * np.pi
            xy = (self.coords[i] - self.coord_mean) / self.coord_std
            out.append(
                [
                    np.sin(a),
                    np.cos(a),
                    np.sin(b),
                    np.cos(b),
                    np.log1p(dt / 3600),
                    float(k == 0),
                    *xy,
                    np.log1p(dist),
                ]
            )
        return np.asarray(out, dtype=np.float32).reshape(-1, 9)

    def audit(self):
        return dict(
            **{k: v for k, v in self.identity.items() if k != "catalog"},
            catalog_size=len(self.catalog),
            normalization="predictor-fit only",
            request_time="last observed timestamp",
            graph_cutoff="strictly before min(query time, predictor-fit cutoff)",
            split_counts={
                s: dict(
                    requests=len(self.split(s)),
                    in_catalog=len(self.split(s, True)),
                    users=len({r.user for r in self.split(s)}),
                )
                for s in ["fit", "policy", "val", "test"]
            },
        )
