"""Fetch a pinned official baseline into ignored external working storage.
This intentionally does not pretend to supply a common-protocol training adapter.
"""

import argparse
from pathlib import Path
import subprocess
from drlpoi.util import read_json

p = argparse.ArgumentParser()
p.add_argument("model", choices=["getnext", "dchl", "ipcm"])
a = p.parse_args()
meta = read_json("configs/upstream.json")[a.model]
path = Path("artifacts/upstream") / a.model
if path.exists():
    raise FileExistsError(path)
subprocess.run(["git", "clone", "--no-checkout", meta["repository"], str(path)], check=True)
subprocess.run(["git", "-C", str(path), "checkout", "--detach", meta["commit"]], check=True)
print("Fetched official source. Audit its license and build a common-protocol adapter; see docs/external.md.")
