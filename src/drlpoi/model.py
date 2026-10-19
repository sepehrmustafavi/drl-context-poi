from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn


def pack(store, requests, masks, device, block_cache=None):
    c = store.cfg
    b, j, l, k = len(requests), len(store.sources), c["recent"], c["block_size"]
    p = np.zeros((b, l), dtype=np.int64)
    cat = p.copy()
    f = np.zeros((b, l, 9), np.float32)
    sp = np.zeros((b, j, k), dtype=np.int64)
    sc = sp.copy()
    sf = np.zeros((b, j, k, 9), np.float32)
    for row, r in enumerate(requests):
        ids = r.prefix[-l:]
        p[row, : len(ids)] = store.p[list(ids)]
        cat[row, : len(ids)] = store.cat[list(ids)]
        f[row, : len(ids)] = store.features(ids)
        for s in range(j):
            if masks[row, s]:
                key = (r.id, s)
                if block_cache is not None and key in block_cache:
                    ids = block_cache[key]
                else:
                    ids = store.retrieve(r, s)
                    if block_cache is not None:
                        block_cache[key] = ids
                sp[row, s, : len(ids)] = store.p[ids]
                sc[row, s, : len(ids)] = store.cat[ids]
                sf[row, s, : len(ids)] = store.features(ids)
    return {
        key: torch.as_tensor(v, device=device)
        for key, v in dict(
            p=p, cat=cat, f=f, sp=sp, sc=sc, sf=sf, selected=np.asarray(masks, dtype=bool)
        ).items()
    }


class Predictor(nn.Module):
    def __init__(self, npoi, ncat, cfg):
        super().__init__()
        self.cfg, self.npoi = cfg, npoi
        d = cfg["dim"]
        self.poi = nn.Embedding(npoi + 2, d, padding_idx=0)
        self.cat = nn.Embedding(ncat + 1, d, padding_idx=0)
        self.time = nn.Linear(6, d, bias=False)
        self.geo = nn.Linear(3, d, bias=False)
        self.position = nn.Embedding(max(cfg["recent"], cfg["block_size"]), d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(cfg["dropout"])
        if cfg["backbone"] in ("transformer", "sasrec"):
            layer = nn.TransformerEncoderLayer(
                d, cfg["heads"], 4 * d, cfg["dropout"], batch_first=True, norm_first=True, activation="gelu"
            )
            self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        else:
            self.encoder = nn.GRU(
                d,
                d,
                num_layers=cfg["layers"],
                batch_first=True,
                dropout=cfg["dropout"] if cfg["layers"] > 1 else 0,
            )
        self.history_encoders = nn.ModuleDict(
            {s: nn.GRU(d, d, batch_first=True) for s in cfg["sources"] if s in ("remote", "periodic")}
        )
        self.key = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in cfg["sources"]])
        self.value = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in cfg["sources"]])
        self.bias = nn.Parameter(torch.zeros(npoi))
        self.apply(self._initialize)
        nn.init.zeros_(self.poi.weight[-1:])

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx : module.padding_idx + 1])
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)

    def embed(self, p, cat, f, positions=False):
        x = self.poi(p)
        if self.cfg["backbone"] != "sasrec":
            x = x + self.cat(cat) + self.time(f[..., :6]) + self.geo(f[..., 6:])
        if positions:
            x = x + self.position(torch.arange(p.shape[-1], device=p.device))
        return self.dropout(self.norm(x))

    def forward(self, batch):
        p = batch["p"]
        x = self.embed(p, batch["cat"], batch["f"], True)
        if self.cfg["backbone"] in ("transformer", "sasrec"):
            causal = torch.triu(torch.ones(p.shape[1], p.shape[1], device=p.device, dtype=torch.bool), 1)
            x = self.encoder(x, mask=causal, src_key_padding_mask=p == 0)
        else:
            x, _ = self.encoder(x)
        last = (p != 0).sum(1) - 1
        base = x[torch.arange(len(p), device=p.device), last]
        if self.cfg["backbone"] == "sasrec":
            return base
        vectors = []
        available = []
        for j, s in enumerate(self.cfg["sources"]):
            ids = batch["sp"][:, j]
            valid = ids != 0
            available.append(valid.any(1) & batch["selected"][:, j])
            if not batch["selected"][:, j].any():
                vectors.append(torch.zeros_like(base))
                continue
            z = self.embed(ids, batch["sc"][:, j], batch["sf"][:, j])
            if s in self.history_encoders:
                z, _ = self.history_encoders[s](z)
            score = torch.einsum("bkd,bd->bk", z, base) / math.sqrt(base.shape[-1])
            safe = valid.clone()
            safe[:, 0] |= ~valid.any(1)
            score = score.masked_fill(~safe, -1e9)
            vec = torch.einsum("bk,bkd->bd", score.softmax(-1), z)
            vectors.append(vec)
        vectors = torch.stack(vectors, 1)
        keys = torch.stack([layer(vectors[:, j]) for j, layer in enumerate(self.key)], 1)
        values = torch.stack([layer(vectors[:, j]) for j, layer in enumerate(self.value)], 1)
        valid = torch.stack(available, 1)
        safe = valid.clone()
        safe[:, 0] |= ~valid.any(1)
        scores = torch.einsum("bjd,bd->bj", keys, base) / math.sqrt(base.shape[-1])
        weights = scores.masked_fill(~safe, -1e9).softmax(-1) * valid
        return base + torch.einsum("bj,bjd->bd", weights, values)

    def logits(self, z, chunk=None):
        chunk = chunk or self.npoi
        return torch.cat(
            [
                z @ self.poi.weight[1 + a : 1 + min(a + chunk, self.npoi)].T + self.bias[a : a + chunk]
                for a in range(0, self.npoi, chunk)
            ],
            -1,
        )


@torch.no_grad()
def ensemble_predict(models, batch, chunk=4096):
    probs, latents, entropy = [], [], []
    for model in models:
        z = model(batch).float()
        p = model.logits(z, chunk).float().softmax(-1)
        probs.append(p)
        latents.append(z)
        entropy.append(-(p * p.clamp_min(1e-30).log()).sum(-1))
    p = torch.stack(probs).mean(0)
    h = -(p * p.clamp_min(1e-30).log()).sum(-1)
    u = (h - torch.stack(entropy).mean(0)).clamp_min(0)
    return p, torch.stack(latents).mean(0), h, u
