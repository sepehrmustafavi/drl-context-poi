from __future__ import annotations
import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch
import yaml


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as f:
        tmp = Path(f.name)
        f.write(payload)
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config(path, overrides=()):
    c = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        k, v = override.split("=", 1)
        if k not in c:
            raise ValueError(f"Unknown configuration key: {k}")
        c[k] = yaml.safe_load(v)
    if c["backbone"] not in ("transformer", "gru", "sasrec"):
        raise ValueError("Unsupported backbone")
    if c["ensemble"] < 1 or c["dim"] % c["heads"]:
        raise ValueError("Invalid ensemble or attention dimensions")
    return c


def seed_everything(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def device_of(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def environment():
    import psutil

    try:
        git = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git = None
    cpu_model = platform.processor() or platform.machine()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    return dict(
        torch_threads=torch.get_num_threads(),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cpu=cpu_model,
        cpu_logical=psutil.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,
        cudnn=torch.backends.cudnn.version(),
        numerical_precision="training AMP FP16 when enabled on CUDA; evaluation FP32",
        python=platform.python_version(),
        platform=platform.platform(),
        torch=torch.__version__,
        numpy=np.__version__,
        cuda=torch.version.cuda,
        git=git,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
    )


def save_torch(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_torch(path, device="cpu"):
    # Load only locally produced, trusted checkpoints.
    return torch.load(path, map_location=device, weights_only=False)
