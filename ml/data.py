"""PaySim validation and chronological splits with whole time-step groups."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_FEATURES = ["step", "type", "amount", "oldbalanceOrg"]
REQUIRED_COLUMNS = [*RAW_FEATURES, "isFraud"]
TRANSACTION_TYPES = ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_features(df: pd.DataFrame) -> None:
    missing = set(RAW_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty")
    for column in ("step", "amount", "oldbalanceOrg"):
        try:
            values = df[column].to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{column} must contain numeric values") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(
                f"{column} must contain finite, non-negative values; check missing data"
            )
    if (df["step"].astype(float) % 1 != 0).any():
        raise ValueError("step must contain integers")
    if not df["type"].isin(TRANSACTION_TYPES).all():
        raise ValueError(f"type must belong to {TRANSACTION_TYPES}")


def validate_dataset(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    validate_features(df)
    if not df["isFraud"].isin([0, 1]).all():
        raise ValueError("isFraud must contain only 0 and 1, with no missing values")
    if df["isFraud"].nunique() != 2:
        raise ValueError("Dataset must contain both classes (0 and 1)")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"Dataset not found: {path}. Download PaySim or use ml.demo_data.")
    try:
        # Read only approved fields and label; sender identifiers never enter memory.
        df = pd.read_csv(path, usecols=lambda column: column in REQUIRED_COLUMNS)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc
    validate_dataset(df)
    return df


def eda_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "inspected_columns": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "duplicate_feature_label_rows": int(df.duplicated().sum()),
        "duplicate_note": "Duplicates measured on loaded columns only; no rows dropped.",
        "class_counts": {str(k): int(v) for k, v in df.isFraud.value_counts().items()},
        "fraud_rate": float(df.isFraud.mean()),
        "transaction_types": {str(k): int(v) for k, v in df.type.value_counts().items()},
        "amount_summary": df.amount.describe().to_dict(),
    }


def partition_summary(frame: pd.DataFrame) -> dict:
    return {
        "rows": len(frame),
        "positive_count": int(frame.isFraud.sum()),
        "negative_count": int((frame.isFraud == 0).sum()),
        "fraud_rate": float(frame.isFraud.mean()),
        "step_min": int(frame.step.min()),
        "step_max": int(frame.step.max()),
    }


def chronological_split(df: pd.DataFrame, checksum: str) -> tuple[dict, dict]:
    """Choose boundaries nearest 70/85% cumulative rows, keeping steps together."""
    validate_dataset(df)
    ordered = df.sort_values("step", kind="stable")
    counts = ordered.groupby("step", sort=True).size()
    if len(counts) < 3:
        raise ValueError("Need at least three distinct steps for train/validation/test")
    cumulative = counts.cumsum().to_numpy()
    train_index = int(np.argmin(abs(cumulative[:-2] - len(df) * 0.70)))
    candidates = np.arange(train_index + 1, len(counts) - 1)
    val_index = int(candidates[np.argmin(abs(cumulative[candidates] - len(df) * 0.85))])
    train_end, validation_end = int(counts.index[train_index]), int(counts.index[val_index])
    parts = {
        "train": ordered[ordered.step <= train_end].copy(),
        "validation": ordered[(ordered.step > train_end) & (ordered.step <= validation_end)].copy(),
        "test": ordered[ordered.step > validation_end].copy(),
    }
    for name, frame in parts.items():
        if frame.isFraud.nunique() != 2:
            raise ValueError(
                f"{name} split must contain both classes, including fraud positives. "
                "Inspect chronological coverage; random fallback is not allowed."
            )
    summary = {
        "schema_version": "1.0",
        "dataset_sha256": checksum,
        "total_rows": len(df),
        "method": "chronological_whole_step_nearest_row_fraction_70_15_15",
        "train_end_step": train_end,
        "validation_end_step": validation_end,
        "partitions": {name: partition_summary(frame) for name, frame in parts.items()},
    }
    return parts, summary


def reconstruct_test(df: pd.DataFrame, summary: dict, checksum: str) -> pd.DataFrame:
    if checksum != summary["dataset_sha256"] or len(df) != summary["total_rows"]:
        raise ValueError("Dataset checksum/row count differs from saved split; evaluation stopped")
    test = df[df.step > summary["validation_end_step"]].sort_values("step", kind="stable")
    if test.empty or test.isFraud.nunique() != 2:
        raise ValueError("Saved test partition must contain both classes")
    if partition_summary(test) != summary["partitions"]["test"]:
        raise ValueError("Test partition does not match saved split metadata")
    return test
