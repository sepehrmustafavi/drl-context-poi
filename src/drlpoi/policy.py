from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch import nn
from .cache import Cache, bits
from .metrics import per_request, summarize
from .util import device_of, seed_everything, save_torch, load_torch, write_json, digest

LEARNED = ("ddqn", "dqn", "no_stop", "no_uncertainty", "no_representation", "myopic", "direct", "bandit")


class QNetwork(nn.Module):
    def __init__(self, input_dim, hidden, actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, x):
        return self.net(x)


def drop_for(method):
    return {"no_uncertainty": "uncertainty", "no_representation": "representation"}.get(method)


def best_fixed(cache):
    # A global subset intersected with request availability; affordable intersection only.
    best = (float("-inf"), float("-inf"), 0)
    for m in range(cache.s):
        if (
            m.bit_count() > cache.cfg["max_acquisitions"]
            or cache.costs[bits(m, cache.j)].sum() > cache.budget + 1e-6
        ):
            continue
        rows = []
        for i in range(len(cache.requests)):
            actual = m & sum((1 << j) for j in range(cache.j) if cache.available[i, j])
            if actual not in cache.feasible(i):
                actual = 0
            rank = int(cache.rank[i, actual])
            score = per_request(rank if rank > 0 else None)["ndcg@10"]
            rows.append((score, float(cache.costs[bits(actual, cache.j)].sum())))
        item = (float(np.mean([r[0] for r in rows])), -float(np.mean([r[1] for r in rows])), m)
        if item[:2] > best[:2]:
            best = item
    return best[2]


def rollout(cache, i, method, network=None, rng=None, fixed=0):
    rng = rng or np.random.default_rng(cache.cfg["seed"])
    feasible = cache.feasible(i)
    if method in ("base", "all", "fixed", "oracle", "direct", "bandit"):
        if method == "base":
            m = 0
        elif method == "all":
            # All available context is an unconstrained quality reference.
            m = sum(1 << j for j in range(cache.j) if cache.available[i, j])
        elif method == "fixed":
            m = fixed & sum(1 << j for j in range(cache.j) if cache.available[i, j])
            if m not in feasible:
                m = 0
        elif method == "oracle":
            if not cache.requests[i]["in_catalog"]:
                raise ValueError("Oracle diagnostic is defined only on in-catalog requests")
            m = max(feasible, key=lambda x: (cache.objective(i, x), -x))
        else:
            with torch.no_grad():
                q = (
                    network(torch.tensor(cache.state(i, 0), device=next(network.parameters()).device))
                    .cpu()
                    .numpy()
                )
            m = max(feasible, key=lambda x: (q[x], -x))
        return m, [j for j in range(cache.j) if (m >> j) & 1] + [cache.j]
    mask = 0
    trace = []
    while True:
        valid = cache.valid(i, mask, method == "no_stop")
        actions = np.flatnonzero(valid)
        if method == "random":
            a = int(rng.choice(actions))
        elif method == "entropy":
            entropy = float(cache.entropy[i, mask]) / max(np.log(cache.meta["catalog_size"]), 1)
            choices = [a for a in cache.cfg["fixed_order"] if valid[a]]
            a = choices[0] if choices and entropy > cache.cfg["entropy_threshold"] else cache.j
        elif method.startswith("leave_"):
            banned = cache.meta["sources"].index(method[6:])
            choices = [a for a in range(cache.j) if a != banned and valid[a]]
            a = choices[0] if choices else cache.j
        else:
            with torch.no_grad():
                q = network(
                    torch.tensor(
                        cache.state(i, mask, drop_for(method)), device=next(network.parameters()).device
                    )
                )
                q = q.cpu().numpy()
            a = int(np.argmax(np.where(valid, q, -np.inf)))
        trace.append(a)
        if a == cache.j:
            return mask, trace
        mask |= 1 << a


@torch.no_grad()
def batched_rollout(cache, method, network):
    """Batch cached policy evaluation; serving latency remains single-request and raw."""
    device = next(network.parameters()).device
    n = len(cache.requests)
    masks = np.zeros(n, dtype=np.int32)
    traces = [[] for _ in range(n)]
    size = cache.cfg.get("policy_eval_batch_size", 512)
    for start in range(0, n, size):
        active = list(range(start, min(start + size, n)))
        for _ in range(cache.j + 1):
            if not active:
                break
            states = np.stack([cache.state(i, int(masks[i]), drop_for(method)) for i in active])
            q = network(torch.tensor(states, device=device)).cpu().numpy()
            if method in ("direct", "bandit"):
                for row, i in enumerate(active):
                    m = max(cache.feasible(i), key=lambda m: (q[row, m], -m))
                    masks[i] = m
                    traces[i] = [j for j in range(cache.j) if (m >> j) & 1] + [cache.j]
                break
            remaining = []
            for row, i in enumerate(active):
                valid = cache.valid(i, int(masks[i]), method == "no_stop")
                a = int(np.argmax(np.where(valid, q[row], -np.inf)))
                traces[i].append(a)
                if a != cache.j:
                    masks[i] |= 1 << a
                    remaining.append(i)
            active = remaining
    return [(int(m), t) for m, t in zip(masks, traces)]


def evaluate(cache, method, network=None, fixed=0):
    rng = np.random.default_rng(cache.cfg["seed"])
    rows = []
    decisions = batched_rollout(cache, method, network) if method in LEARNED and network is not None else None
    for i, r in enumerate(cache.requests):
        if method == "oracle" and not r["in_catalog"]:
            continue
        mask, trace = (
            decisions[i] if decisions is not None else rollout(cache, i, method, network, rng, fixed)
        )
        rank = int(cache.rank[i, mask])
        metrics = per_request(
            rank if rank > 0 else None, float(cache.reward_nll[i, mask]) if r["in_catalog"] else None
        )
        rows.append(
            dict(
                request_id=r["id"],
                user=r["user"],
                in_catalog=r["in_catalog"],
                mask=mask,
                cost=float(cache.costs[bits(mask, cache.j)].sum()),
                acquisitions=mask.bit_count(),
                trace=trace,
                **metrics,
            )
        )
    return rows, summarize(rows)


def make_transition(cache, i, mask, action, force_stop=False, drop=None):
    state = cache.state(i, mask, drop)
    if not cache.valid(i, mask, force_stop)[action]:
        raise ValueError("Illegal action")
    done = action == cache.j
    new = mask if done else mask | (1 << action)
    reward = -float(cache.reward_nll[i, mask]) if done else -cache.cfg["lambda_cost"] * cache.costs[action]
    return (
        state,
        action,
        float(reward),
        cache.state(i, new, drop),
        done,
        cache.valid(i, new, force_stop),
    ), new


def td_targets(rewards, dones, next_online, next_target, valid, double=True):
    q = next_online if double else next_target
    best = q.masked_fill(~valid, float("-inf")).argmax(-1)
    values = next_target.gather(1, best[:, None]).squeeze(1)
    return rewards + (~dones).float() * values  # gamma=1, explicit terminal STOP


def train(train_path, val_path, cfg, method, out, resume=False):
    if method not in LEARNED:
        raise ValueError("Not a learned policy")
    tr, va = Cache(train_path, cfg), Cache(val_path, cfg)
    tr.compatible(va)
    if tr.meta["identity"]["split"] != "policy" or va.meta["identity"]["split"] != "val":
        raise ValueError("Train controller on policy-fit requests and select on validation")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "policy.pt").exists() and not resume:
        raise FileExistsError("Use --resume or a fresh policy output directory")
    seed_everything(cfg["seed"], cfg["threads"])
    rng = np.random.default_rng(cfg["seed"])
    dev = device_of(cfg["device"])
    actions = tr.s if method in ("direct", "bandit") else tr.j + 1
    net = QNetwork(tr.state_dim, cfg["q_hidden"], actions).to(dev)
    target = QNetwork(tr.state_dim, cfg["q_hidden"], actions).to(dev)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg["q_lr"])
    replay = deque(maxlen=cfg["replay_capacity"])
    best = (-np.inf, -np.inf)
    updates = 0
    total_steps = cfg["policy_epochs"] * cfg["policy_steps_per_epoch"]
    elapsed = 0
    history = []
    start = 0
    signature = digest(
        dict(
            config={k: v for k, v in cfg.items() if k not in ("device", "threads")},
            method=method,
            train=tr.meta["signature"],
            validation=va.meta["signature"],
        )
    )
    last = out / "last.pt"
    if resume and last.exists():
        ck = load_torch(last, dev)
        if ck["signature"] != signature:
            raise ValueError("Policy resume configuration/cache mismatch")
        net.load_state_dict(ck["model"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["optimizer"])
        replay.extend(ck["replay"])
        rng.bit_generator.state = ck["numpy_rng"]
        start = ck["epoch"] + 1
        best = tuple(ck["best"])
        updates = ck["updates"]
        elapsed = ck["elapsed"]
        history = ck["history"]
    for epoch in range(start, cfg["policy_epochs"]):
        net.train()
        losses = []
        used = 0
        while used < cfg["policy_steps_per_epoch"]:
            i = int(rng.integers(len(tr.requests)))
            mask = 0
            if method in ("direct", "bandit", "myopic"):
                if method == "myopic":
                    feasible = tr.feasible(i)
                    mask = int(rng.choice(feasible))
                state = tr.state(i, mask, drop_for(method))
                x = torch.tensor(state, device=dev)[None]
                q = net(x)
                if method == "direct":
                    feasible = tr.feasible(i)
                    truth = max(feasible, key=lambda m: (tr.objective(i, m), -m))
                    valid = torch.zeros(tr.s, dtype=torch.bool, device=dev)
                    valid[list(feasible)] = True
                    loss = nn.functional.cross_entropy(
                        q.masked_fill(~valid[None], -1e9), torch.tensor([truth], device=dev)
                    )
                elif method == "bandit":
                    feasible = tr.feasible(i)
                    eps = cfg["epsilon_end"] + (cfg["epsilon_start"] - cfg["epsilon_end"]) * max(
                        0, 1 - elapsed / total_steps
                    )
                    action = (
                        int(rng.choice(feasible))
                        if rng.random() < eps
                        else max(feasible, key=lambda m: q[0, m].item())
                    )
                    # Only the selected subset reward is used in this bandit update.
                    loss = nn.functional.smooth_l1_loss(
                        q[0, action], torch.tensor(tr.objective(i, action), device=dev)
                    )
                else:
                    valid = tr.valid(i, mask)
                    truths = np.zeros(tr.j + 1, np.float32)
                    for a in np.flatnonzero(valid[:-1]):
                        truths[a] = (
                            float(tr.reward_nll[i, mask] - tr.reward_nll[i, mask | (1 << int(a))])
                            - cfg["lambda_cost"] * tr.costs[a]
                        )
                    vm = torch.tensor(valid, device=dev)
                    loss = nn.functional.smooth_l1_loss(q[0, vm], torch.tensor(truths, device=dev)[vm])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5)
                opt.step()
                losses.append(loss.item())
                used += 1
                elapsed += 1
                continue
            while True:
                eps = cfg["epsilon_end"] + (cfg["epsilon_start"] - cfg["epsilon_end"]) * max(
                    0, 1 - elapsed / total_steps
                )
                valid = tr.valid(i, mask, method == "no_stop")
                with torch.no_grad():
                    q = net(torch.tensor(tr.state(i, mask, drop_for(method)), device=dev)).cpu().numpy()
                action = (
                    int(rng.choice(np.flatnonzero(valid)))
                    if rng.random() < eps
                    else int(np.argmax(np.where(valid, q, -np.inf)))
                )
                transition, new = make_transition(tr, i, mask, action, method == "no_stop", drop_for(method))
                replay.append(transition)
                used += 1
                elapsed += 1
                if len(replay) >= max(cfg["replay_warmup"], cfg["policy_batch_size"]):
                    indices = rng.choice(len(replay), size=cfg["policy_batch_size"], replace=False)
                    batch = [replay[int(k)] for k in indices]
                    st, ac, re, ns, done, vm = zip(*batch)
                    st = torch.tensor(np.stack(st), device=dev)
                    ns = torch.tensor(np.stack(ns), device=dev)
                    ac = torch.tensor(ac, device=dev)
                    re = torch.tensor(re, device=dev, dtype=torch.float32)
                    done = torch.tensor(done, device=dev)
                    vm = torch.tensor(np.stack(vm), device=dev)
                    with torch.no_grad():
                        targets = td_targets(re, done, net(ns), target(ns), vm, method != "dqn")
                    loss = nn.functional.smooth_l1_loss(net(st).gather(1, ac[:, None]).squeeze(1), targets)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5)
                    opt.step()
                    updates += 1
                    losses.append(loss.item())
                    if updates % cfg["target_interval"] == 0:
                        target.load_state_dict(net.state_dict())
                mask = new
                if transition[4]:
                    break
        net.eval()
        _, summary = evaluate(va, method, net)
        score = (summary["all"]["request_macro"]["ndcg@10"], -summary["cost_mean"])
        if score > best:
            best = score
            save_torch(
                out / "policy.pt",
                dict(
                    model=net.state_dict(),
                    method=method,
                    config=cfg,
                    input_dim=tr.state_dim,
                    actions=actions,
                    train_cache=tr.meta["signature"],
                    predictor_identity=tr.meta["identity"]["model_hashes"],
                    data_identity=tr.meta["identity"]["data"],
                    epoch=epoch,
                    val=summary,
                ),
            )
        row = dict(epoch=epoch, loss=float(np.mean(losses)) if losses else None, validation=summary)
        history.append(row)
        write_json(out / "history.json", history)
        save_torch(
            last,
            dict(
                signature=signature,
                model=net.state_dict(),
                target=target.state_dict(),
                optimizer=opt.state_dict(),
                replay=list(replay),
                numpy_rng=rng.bit_generator.state,
                epoch=epoch,
                best=best,
                updates=updates,
                elapsed=elapsed,
                history=history,
            ),
        )
        print(f"policy={method} epoch={epoch} val_ndcg={score[0]:.5f} cost={-score[1]:.4f}", flush=True)
    write_json(out / "config.json", cfg)


def load_policy(path, cache):
    ck = load_torch(path, device_of(cache.cfg["device"]))
    if (
        ck["predictor_identity"] != cache.meta["identity"]["model_hashes"]
        or ck["data_identity"] != cache.meta["identity"]["data"]
    ):
        raise ValueError("Policy/predictor/data mismatch")
    for k in ("costs", "lambda_cost", "budget_fraction", "max_acquisitions", "disabled_sources"):
        if ck["config"][k] != cache.cfg[k]:
            raise ValueError(f"Policy {k} differs: retrain policy for sensitivity experiment")
    net = QNetwork(ck["input_dim"], ck["config"]["q_hidden"], ck["actions"]).to(
        device_of(cache.cfg["device"])
    )
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck["method"]
