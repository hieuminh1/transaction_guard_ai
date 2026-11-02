from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def select_threshold(y_validation, scores, minimum_precision: float = 0.2) -> dict:
    """Maximize validation recall under precision constraint, then precision/threshold."""
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1]")
    precision, recall, thresholds = precision_recall_curve(y_validation, scores)
    # Last PR point has no threshold; exclude it from the policy search.
    p, r = precision[:-1], recall[:-1]
    eligible = np.flatnonzero(p >= minimum_precision)
    met = bool(len(eligible))
    if met:
        index = max(eligible, key=lambda i: (r[i], p[i], thresholds[i]))
        policy = "max_recall_at_minimum_precision"
    else:
        f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
        index = int(np.argmax(f1))
        policy = "fallback_best_validation_f1"
    return {
        "threshold": float(thresholds[index]),
        "policy": policy,
        "minimum_precision": minimum_precision,
        "precision_target_met": met,
        "selection_split": "validation",
    }


def classification_metrics(y_true, scores, threshold: float) -> dict:
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(y_true),
        "fraud_prevalence": float(np.mean(y_true)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_evaluation_plots(y_true, scores, threshold: float, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, np.asarray(scores) >= threshold, labels=[0, 1], ax=ax, colorbar=False
    )
    ax.set_title("Offline held-out test confusion matrix")
    fig.savefig(output / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label="Held-out test")
    ax.axhline(np.mean(y_true), linestyle="--", label="Fraud prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title="Offline test precision–recall curve")
    ax.legend()
    fig.savefig(output / "precision_recall_curve.png", bbox_inches="tight")
    plt.close(fig)
