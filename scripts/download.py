"""Download a user-selected official dataset URL and record its checksum."""

import argparse
import hashlib
from pathlib import Path
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--out", required=True)
p.add_argument("--sha256")
a = p.parse_args()
dest = Path(a.out)
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    raise FileExistsError(dest)
tmp = dest.with_suffix(dest.suffix + ".part")
h = hashlib.sha256()
try:
    with urllib.request.urlopen(a.url, timeout=60) as response, open(tmp, "wb") as f:
        while block := response.read(1024 * 1024):
            f.write(block)
            h.update(block)
    if a.sha256 and h.hexdigest() != a.sha256:
        raise ValueError("Checksum mismatch")
    tmp.replace(dest)
finally:
    tmp.unlink(missing_ok=True)
print(h.hexdigest(), dest)
