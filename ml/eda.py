"""Descriptive plots only; full-dataset summaries never enter model fitting."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ml.data import eda_summary, load_dataset, write_json


def analyze(data: Path, output: Path) -> dict:
    frame = load_dataset(data)
    summary = eda_summary(frame)
    write_json(output / "eda_summary.json", summary)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    frame.isFraud.value_counts().sort_index().plot.bar(ax=axes[0], logy=True)
    axes[0].set(title="Class imbalance", xlabel="isFraud", ylabel="Transactions (log scale)")
    frame.type.value_counts().plot.bar(ax=axes[1])
    axes[1].set(title="Transaction types", xlabel="Type", ylabel="Transactions")
    axes[2].hist(np.log1p(frame.amount), bins=60)
    axes[2].set(title="Amount distribution", xlabel="log(1 + amount)", ylabel="Transactions")
    fig.tight_layout()
    fig.savefig(output / "eda_distributions.png", bbox_inches="tight")
    plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    try:
        print(analyze(args.data, args.output))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"EDA failed: {exc}\n")


if __name__ == "__main__":
    main()
