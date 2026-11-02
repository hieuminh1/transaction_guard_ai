import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score

from ml.metrics import classification_metrics, select_threshold


def test_known_metrics():
    labels, scores = [0, 0, 1, 1], [0.1, 0.9, 0.3, 0.8]
    result = classification_metrics(labels, scores, 0.5)
    assert [result[k] for k in ("tn", "fp", "fn", "tp")] == [1, 1, 1, 1]
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["average_precision"] == average_precision_score(labels, scores)


def test_validation_threshold_matches_brute_force():
    labels, scores = np.array([0, 1, 0, 1, 0]), np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    result = select_threshold(labels, scores, minimum_precision=0.5)
    eligible = [s for s in scores if precision_score(labels, scores >= s) >= 0.5]
    best_recall = max(recall_score(labels, scores >= s) for s in eligible)
    assert recall_score(labels, scores >= result["threshold"]) == best_recall
    assert result["precision_target_met"]
    assert result["selection_split"] == "validation"


def test_threshold_fallback_is_explicit():
    result = select_threshold([1, 0], [0.1, 0.9], minimum_precision=0.9)
    assert not result["precision_target_met"]
    assert result["policy"] == "fallback_best_validation_f1"
    with pytest.raises(ValueError):
        select_threshold([1, 0], [0.1, 0.9], 0)
