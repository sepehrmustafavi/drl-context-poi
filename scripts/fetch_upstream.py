"""Fetch and verify a pinned upstream model without running its entry point."""

import argparse
from drlpoi.upstream import fetch

p = argparse.ArgumentParser()
p.add_argument("model", choices=["sasrec", "getnext", "dchl", "ipcm"])
a = p.parse_args()
path, provenance = fetch(a.model)
print(path, provenance)
