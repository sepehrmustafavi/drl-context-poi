from pathlib import Path
import numpy as np
import torch
from .data import Store
from .model import Predictor, pack
from .metrics import rank_of
from .util import seed_everything, device_of, save_torch, load_torch, write_json, environment, digest


def fit(csv, cfg, out, resume=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = Store(csv, cfg)
    device = device_of(cfg["device"])
    write_json(out / "audit.json", store.audit())
    write_json(out / "config.json", cfg)
    write_json(out / "environment.json", environment())
    requests = store.split("fit", True)
    validation = store.split("val", True)
    if not requests or not validation:
        raise ValueError("Empty predictor-fit or validation split")
    signature = digest(dict(data=store.identity, config=cfg))
    for member in range(cfg["ensemble"]):
        seed_everything(cfg["seed"] + member * 1009, cfg["threads"])
        model = Predictor(len(store.catalog), len(store.cmap), cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.0)
        start, best, stale = 0, -1.0, 0
        last = out / f"member_{member}.last.pt"
        bestpath = out / f"member_{member}.pt"
        if bestpath.exists() and not resume:
            raise FileExistsError(f"{bestpath}: use --resume or a fresh output directory")
        rng = np.random.default_rng(cfg["seed"] + member * 1009)
        amp = cfg["amp"] and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        if resume and last.exists():
            ck = load_torch(last, device)
            if ck["signature"] != signature:
                raise ValueError("Resume configuration/data mismatch")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["optimizer"])
            scaler.load_state_dict(ck["scaler"])
            torch.set_rng_state(ck["torch_rng"].cpu())
            if device.type == "cuda" and ck["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])
            rng.bit_generator.state = ck["numpy_rng"]
            start, best, stale = ck["epoch"] + 1, ck["best"], ck["stale"]
        history = []
        for epoch in range(start, cfg["epochs"]):
            if stale >= cfg["patience"]:
                break
            model.train()
            losses = []
            for start_idx in range(0, len(requests), cfg["batch_size"]):
                # One permutation per epoch, not one replacement draw per minibatch.
                if start_idx == 0:
                    order = rng.permutation(len(requests))
                rs = [requests[i] for i in order[start_idx : start_idx + cfg["batch_size"]]]
                masks = rng.random((len(rs), len(store.sources))) < 0.5
                # Include guaranteed empty and full masks as well as random subsets.
                masks[0] = False
                if len(rs) > 1:
                    masks[1] = True
                if cfg["backbone"] == "sasrec":
                    masks[:] = False
                batch = pack(store, rs, masks, device)
                y = torch.tensor([store.target(r) for r in rs], device=device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    logits = model.logits(model(batch), cfg["score_chunk"])
                    loss = torch.nn.functional.cross_entropy(logits.float(), y)
                    # Eq. 21 specifies an explicit L2 term, rather than decoupled weight decay.
                    loss = loss + cfg["weight_decay"] * sum(
                        p.float().square().sum() for p in model.parameters()
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                scaler.step(opt)
                scaler.update()
                losses.append(loss.item())
            model.eval()
            ndcgs = []
            # Predictor checkpoint selection uses validation, never policy-fit or test labels.
            with torch.no_grad():
                for start_idx in range(0, len(validation), cfg["batch_size"]):
                    rs = validation[start_idx : start_idx + cfg["batch_size"]]
                    metrics = []
                    for selected in (False, True):
                        masks = np.full(
                            (len(rs), len(store.sources)), selected and cfg["backbone"] != "sasrec"
                        )
                        batch = pack(store, rs, masks, device)
                        logits = model.logits(model(batch), cfg["score_chunk"]).float().cpu().numpy()
                        metrics.append(
                            [
                                1 / np.log2(rank + 1) if (rank := rank_of(p, store.target(r))) <= 10 else 0
                                for r, p in zip(rs, logits)
                            ]
                        )
                    ndcgs.extend(np.mean(metrics, axis=0).tolist())
            metric = float(np.mean(ndcgs))
            if metric > best:
                best, stale = metric, 0
                save_torch(
                    bestpath,
                    dict(
                        model=model.state_dict(),
                        config=cfg,
                        identity=store.identity,
                        signature=signature,
                        member=member,
                        epoch=epoch,
                        val_ndcg=metric,
                    ),
                )
            else:
                stale += 1
            save_torch(
                last,
                dict(
                    model=model.state_dict(),
                    optimizer=opt.state_dict(),
                    scaler=scaler.state_dict(),
                    signature=signature,
                    epoch=epoch,
                    best=best,
                    stale=stale,
                    numpy_rng=rng.bit_generator.state,
                    torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                ),
            )
            row = dict(epoch=epoch, loss=float(np.mean(losses)), validation_ndcg=metric)
            history.append(row)
            print(f"member={member} {row}", flush=True)
            # Append logs across resumptions.
            with open(out / f"member_{member}.history.jsonl", "a") as f:
                import json

                f.write(json.dumps(row) + "\n")
    return out


def load_ensemble(csv, predictor_dir, device=None):
    p = Path(predictor_dir)
    from .util import read_json

    cfg = read_json(p / "config.json")
    if device:
        cfg["device"] = device
    store = Store(csv, cfg)
    dev = device_of(cfg["device"])
    models = []
    for member in range(cfg["ensemble"]):
        ck = load_torch(p / f"member_{member}.pt", dev)
        if ck["identity"] != store.identity:
            raise ValueError("Predictor/data/index mismatch")
        model = Predictor(len(store.catalog), len(store.cmap), cfg).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        model.requires_grad_(False)
        models.append(model)
    return store, models, dev
