"""Publication-exportable plots of actual completed results; no simulated performance."""

from collections import defaultdict
from pathlib import Path
import numpy as np
from .util import read_json


def figures(aggregate, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = read_json(aggregate)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for name, row in data.items():
        grouped[name.split("/")[0]].append((name, row))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
        }
    )
    for dataset, items in grouped.items():
        # Separate scenarios are displayed explicitly; no averaging across distinct variants.
        fig, ax = plt.subplots(figsize=(max(7, len(items) * 0.35), 4))
        labels = []
        for i, (name, row) in enumerate(items):
            nd = row["metrics"]["ndcg@10"]
            color = "#236B8E" if "/ddqn/" in "/" + name + "/" else "#78A6A3"
            ax.bar(i, nd["mean"], color=color, width=0.7)
            if nd["std"] is not None:
                ax.errorbar(i, nd["mean"], yerr=nd["std"], color="#263238", capsize=3, fmt="none")
            labels.append(name[len(dataset) + 1 :] + f" (n={row['n']})")
        ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right")
        ax.set_ylabel("NDCG@10 (all requests)")
        ax.set_title(dataset + " — completed runs")
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        for ext in ("png", "pdf"):
            fig.savefig(out / f"{dataset}_ndcg.{ext}", dpi=300)
        plt.close(fig)
