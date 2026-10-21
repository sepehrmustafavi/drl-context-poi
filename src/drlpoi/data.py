"""Chronological requests and historical-only, bounded context retrieval."""

from __future__ import annotations
from bisect import bisect_left, bisect_right, insort
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
    elif fmt == "getnext-gowalla":
        import zipfile

        expected = "c5c0604ab2ee68219338541bc23fe7e9f2023c6e9569bafda68488bde87afc3f"
        if file_hash(raw) != expected:
            raise ValueError("Not the pinned GETNext Gowalla-CA archive; source/version mismatch")
        with zipfile.ZipFile(raw) as archive:
            parts = [
                pd.read_csv(archive.open(f"gowalla_{split}.csv"), dtype={"user_id": str, "POI_id": str})
                for split in ("train", "val", "test")
            ]
        df = pd.concat(parts, ignore_index=True).rename(
            columns={"POI_id": "poi_id", "checkin_time": "timestamp"}
        )
        # Metadata acquisition time is undocumented: conservatively disable categories.
        df["category"] = "__missing__"
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
            dataset_name=(
                "Gowalla-CA GETNext release"
                if fmt == "getnext-gowalla"
                else "Foursquare NYC"
                if fmt == "foursquare" and "NYC" in Path(raw).name.upper()
                else "Foursquare Tokyo"
                if fmt == "foursquare" and "TKY" in Path(raw).name.upper()
                else fmt
            ),
            source_url="https://github.com/songyangco/GETNext/blob/6a79081aec6951ebec31bff6400d6ba0e40a613e/dataset/Gowalla-CA.zip"
            if fmt == "getnext-gowalla"
            else "https://sites.google.com/site/yangdingqi/home/foursquare-dataset"
            if fmt == "foursquare"
            else "https://snap.stanford.edu/data/loc-gowalla.html"
            if fmt == "gowalla"
            else None,
            source_version=(
                "6a79081aec6951ebec31bff6400d6ba0e40a613e"
                if fmt == "getnext-gowalla"
                else "TSMC2014; exact raw bytes identified by raw_sha256"
                if fmt == "foursquare"
                else "SNAP loc-gowalla; exact raw bytes and optional bbox recorded"
                if fmt == "gowalla"
                else None
            ),
            inherited_preprocessing="upstream released subset; merge all three supplied splits, discard upstream graphs/splits, disable unverified categories"
            if fmt == "getnext-gowalla"
            else None,
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
        self.df["timestamp"] = pd.to_datetime(self.df.timestamp, utc=True, format="mixed", errors="raise")
        self.df = self.df.sort_values(["timestamp", "user_id", "poi_id"], kind="stable").reset_index(
            drop=True
        )
        self.t = (
            pd.to_datetime(self.df.timestamp, utc=True, format="mixed")
            .dt.as_unit("ns")
            .astype("int64")
            .to_numpy()
            / 1e9
        )
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
        self.cmap = {
            p: i + 1 for i, p in enumerate(sorted(p for p in fit.category.unique() if p != "__missing__"))
        }
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
            if len(h) >= cfg["min_history"] and self.t[h[-1]] < self.t[idx]:
                split = ("fit", "policy", "val", "test")[
                    int(np.searchsorted(self.cuts, self.t[idx], side="right"))
                ]
                self.requests.append(Request(f"r{idx}", row.user_id, Prefix(h, len(h)), idx, split))
            if h and self.t[idx] < self.cuts[0]:
                self.edges[int(self.p[h[-1]])].append((self.t[idx], idx))
                self.cedges[int(self.cat[h[-1]])].append((self.t[idx], idx))
            h.append(idx)
        self.sources = list(cfg["sources"])
        if len(set(self.sources)) != len(self.sources) or not set(self.sources) <= set(SOURCES):
            raise ValueError("Supported sources: " + str(SOURCES))
        self.identity = dict(
            protocol_version=2,
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
        self._periodic_index = defaultdict(list)
        self._periodic_counts = {}
        self._user_positions = {}
        self._previous = {}
        for user, indices in users.items():
            clocks = defaultdict(list)
            inserted = 0
            for pos, idx in enumerate(indices):
                self._user_positions[idx] = pos
                self._previous[idx] = indices[pos - 1] if pos else None
                day = int(self.t[idx] // 86400) % 7
                self._periodic_index[(user, day)].append(idx)
                # Only events outside the recent window enter the count index.
                end = max(0, pos + 1 - cfg["recent"])
                while inserted < end:
                    old = indices[inserted]
                    insort(clocks[int(self.t[old] // 86400) % 7], self.t[old] % 86400)
                    inserted += 1
                times = clocks[day]
                hour = self.t[idx] % 86400
                width = cfg["periodic_hour_window"] * 3600
                if width < 0:
                    raise ValueError("Periodic window must be nonnegative")

                def count(lo, hi):
                    return bisect_right(times, hi) - bisect_left(times, lo)

                if width >= 43200:
                    total = len(times)
                elif hour - width < 0:
                    total = count(0, hour + width) + count(hour - width + 86400, 86400)
                elif hour + width >= 86400:
                    total = count(hour - width, 86400) + count(0, hour + width - 86400)
                else:
                    total = count(hour - width, hour + width)
                self._periodic_counts[idx] = total
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
        end = max(0, len(r.prefix) - self.cfg["recent"])
        if source == "remote":
            return r.prefix[:end]
        tau = self.t[r.prefix[-1]]
        day = int(tau // 86400) % 7
        hour = tau / 3600 % 24
        w = self.cfg["periodic_hour_window"]
        return [
            i
            for i in self._periodic_index[(r.user, day)]
            if self._user_positions[i] < end
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
            n = (
                max(0, len(r.prefix) - self.cfg["recent"])
                if s == "remote"
                else self._periodic_counts[r.prefix[-1]]
                if s == "periodic"
                else self._edge_prefix(r, s)[1]
            )
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
            previous = self._previous[i]
            dt = max(0, t - self.t[previous]) if previous is not None else 0
            dist = haversine(self.coords[i], self.coords[previous]) if previous is not None else 0
            a, b = t / 86400 * 2 * np.pi, t / (86400 * 7) * 2 * np.pi
            xy = (self.coords[i] - self.coord_mean) / self.coord_std
            out.append(
                [
                    np.sin(a),
                    np.cos(a),
                    np.sin(b),
                    np.cos(b),
                    np.log1p(dt / 3600),
                    float(previous is None),
                    *xy,
                    np.log1p(dist),
                ]
            )
        return np.asarray(out, dtype=np.float32).reshape(-1, 9)

    def audit(self):
        def partition(s, lower, upper):
            events = self.df[(self.t >= lower) & (self.t < upper)]
            rs = self.split(s)
            lengths = np.array([len(r.prefix) for r in rs])
            pairs = events[["user_id", "poi_id"]].drop_duplicates()
            denom = events.user_id.nunique() * events.poi_id.nunique()
            return dict(
                events=len(events),
                event_users=events.user_id.nunique(),
                pois=events.poi_id.nunique(),
                requests=len(rs),
                in_catalog=len(self.split(s, True)),
                out_of_catalog=len(rs) - len(self.split(s, True)),
                users=len({r.user for r in rs}),
                sparsity=1 - len(pairs) / denom if denom else None,
                history_quantiles=np.quantile(lengths, [0, 0.25, 0.5, 0.75, 0.95, 1]).tolist()
                if len(lengths)
                else [],
                lower_utc=pd.Timestamp(lower, unit="s", tz="UTC").isoformat(),
                upper_utc_exclusive=pd.Timestamp(upper, unit="s", tz="UTC").isoformat(),
            )

        bounds = [float(self.t.min()), *self.cuts.tolist(), float(self.t.max() + 1)]
        return dict(
            **{k: v for k, v in self.identity.items() if k != "catalog"},
            total_events=len(self.df),
            total_users=self.df.user_id.nunique(),
            total_pois=self.df.poi_id.nunique(),
            sparsity_definition="1 - unique user/POI pairs divided by event-user count times POI count within partition",
            catalog_size=len(self.catalog),
            normalization="predictor-fit only",
            request_time="last observed timestamp",
            graph_cutoff="strictly before min(query time, predictor-fit cutoff)",
            manifest=__import__("json").loads(Path(self.path + ".manifest.json").read_text())
            if Path(self.path + ".manifest.json").exists()
            else None,
            split_counts={
                s: partition(s, bounds[i], bounds[i + 1])
                for i, s in enumerate(["fit", "policy", "val", "test"])
            },
        )
