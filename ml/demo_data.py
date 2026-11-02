"""Generate a small artificial smoke-test CSV. This is NOT the PaySim dataset."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data import TRANSACTION_TYPES


def generate_demo_data(steps: int = 60, rows_per_step: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    count = steps * rows_per_step
    amount = rng.lognormal(6, 2, count)
    balance = rng.lognormal(8, 2, count)
    types = rng.choice(TRANSACTION_TYPES, count)
    fraud = ((amount > balance * 0.8) & np.isin(types, ["TRANSFER", "CASH_OUT"])).astype(int)
    # Guarantee both labels at each step solely for smoke-test split coverage.
    fraud[::rows_per_step] = 1
    fraud[1::rows_per_step] = 0
    return pd.DataFrame(
        {
            "step": np.repeat(np.arange(steps), rows_per_step),
            "type": types,
            "amount": amount,
            "oldbalanceOrg": balance,
            "isFraud": fraud,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/demo.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data().to_csv(args.output, index=False)
    print(f"Artificial smoke-test data written to {args.output}; not PaySim performance evidence.")


if __name__ == "__main__":
    main()
