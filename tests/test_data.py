import numpy as np
import pytest

from ml.data import chronological_split, load_dataset, reconstruct_test, validate_dataset


@pytest.mark.parametrize("column", ["isFraud", "amount", "step", "type", "oldbalanceOrg"])
def test_missing_columns(frame, column):
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(frame.drop(columns=column))


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf])
def test_invalid_amount(frame, value):
    frame.loc[0, "amount"] = value
    with pytest.raises(ValueError, match="amount"):
        validate_dataset(frame)


def test_split_disjoint_reproducible(frame):
    parts, summary = chronological_split(frame, "checksum")
    assert sum(len(part) for part in parts.values()) == len(frame)
    for left, right in [("train", "validation"), ("validation", "test"), ("train", "test")]:
        assert set(parts[left].index).isdisjoint(parts[right].index)
        assert set(parts[left].step).isdisjoint(parts[right].step)
        assert parts[left].step.max() < parts[right].step.min()
    restored = reconstruct_test(frame, summary, "checksum")
    assert restored.equals(parts["test"])
    with pytest.raises(ValueError, match="checksum"):
        reconstruct_test(frame, summary, "different")


def test_no_test_positives_stops(frame):
    frame.loc[frame.step >= 17, "isFraud"] = 0
    with pytest.raises(ValueError, match="test split must contain both classes"):
        chronological_split(frame, "checksum")


def test_read_csv_only_approved_fields(frame, tmp_path):
    frame["nameOrig"] = "must-not-load"
    path = tmp_path / "test.csv"
    frame.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert "nameOrig" not in loaded
    assert len(loaded) == len(frame)


def test_split_keeps_uneven_step_groups(frame):
    frame = frame.loc[~((frame.step % 2 == 0) & (frame.index % 10 > 3))]
    parts, _ = chronological_split(frame, "checksum")
    assert sum(len(part) for part in parts.values()) == len(frame)
    assert parts["validation"].step.min() > parts["train"].step.max()
