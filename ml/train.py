"""Train two supervised models without using the test set for selection."""

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.artifact import save_artifact
from ml.data import (
    RAW_FEATURES,
    chronological_split,
    eda_summary,
    file_sha256,
    load_dataset,
    write_json,
)
from ml.metrics import classification_metrics, save_evaluation_plots, select_threshold
from ml.preprocessing import build_pipeline


def train(
    data: Path,
    output: Path,
    reports: Path,
    config: dict,
    dataset_name: str = "PaySim synthetic",
    tracking_uri: str | None = None,
) -> dict:
    reports.mkdir(parents=True, exist_ok=True)
    df = load_dataset(data)
    checksum = file_sha256(data)
    write_json(reports / "eda_summary.json", eda_summary(df))
    parts, split = chronological_split(df, checksum)
    write_json(reports / "split_summary.json", split)
    seed = int(config["seed"])
    candidates = [
        (
            "LogisticRegression",
            build_pipeline(LogisticRegression(random_state=seed, **config["logistic_regression"])),
        ),
        (
            "RandomForestClassifier",
            build_pipeline(
                RandomForestClassifier(random_state=seed, **config["random_forest"]), scale=False
            ),
        ),
    ]
    comparison, fitted = [], {}
    for name, pipeline in candidates:
        start = time.perf_counter()
        pipeline.fit(parts["train"][RAW_FEATURES], parts["train"].isFraud)
        fit_seconds = time.perf_counter() - start
        scores = pipeline.predict_proba(parts["validation"][RAW_FEATURES])[:, 1]
        threshold = select_threshold(
            parts["validation"].isFraud, scores, float(config["minimum_precision"])
        )
        validation = classification_metrics(
            parts["validation"].isFraud, scores, threshold["threshold"]
        )
        baseline = classification_metrics(parts["validation"].isFraud, scores, 0.5)
        comparison.append(
            {
                "algorithm": name,
                "split": "validation",
                "fit_seconds": fit_seconds,
                **validation,
                **{f"baseline_0_5_{k}": baseline[k] for k in ("precision", "recall", "f1")},
                "parameters": json.dumps(pipeline.named_steps["classifier"].get_params()),
            }
        )
        fitted[name] = (pipeline, threshold, validation)
    best_ap = max(row["average_precision"] for row in comparison)
    tolerance = float(config.get("ap_tie_tolerance", 0.001))
    # Candidate order explicitly prefers the simpler baseline for near ties.
    winner = next(row for row in comparison if row["average_precision"] >= best_ap - tolerance)
    selected_name = winner["algorithm"]
    pipeline, threshold, validation = fitted[selected_name]
    pd.DataFrame(comparison).to_csv(reports / "model_comparison.csv", index=False)
    write_json(
        reports / "validation_metrics.json",
        {
            "algorithm": selected_name,
            "split": "validation",
            **validation,
            "threshold_selection": threshold,
            "selection_rationale": f"Highest validation AP; prefer LR within {tolerance} AP.",
        },
    )
    # Model and threshold are frozen above. This is the only test scoring pass in training.
    start = time.perf_counter()
    test_scores = pipeline.predict_proba(parts["test"][RAW_FEATURES])[:, 1]
    batch_latency_ms = (time.perf_counter() - start) * 1000
    test_metrics = {
        "algorithm": selected_name,
        "split": "final_test",
        **classification_metrics(parts["test"].isFraud, test_scores, threshold["threshold"]),
        "vectorized_batch_latency_ms": batch_latency_ms,
        "amortized_latency_ms_per_row": batch_latency_ms / len(parts["test"]),
    }
    # Warm, repeated single-row timings measure inference only, not HTTP/database time.
    row = parts["test"][RAW_FEATURES].iloc[:1]
    pipeline.predict_proba(row)
    timings = []
    for _ in range(20):
        start = time.perf_counter()
        pipeline.predict_proba(row)
        timings.append((time.perf_counter() - start) * 1000)
    test_metrics["single_row_latency_ms"] = {
        "samples": len(timings),
        "p50": float(np.percentile(timings, 50)),
        "p95": float(np.percentile(timings, 95)),
    }
    now = datetime.now(UTC)
    metadata = {
        "schema_version": "1.0",
        "model_version": now.strftime("%Y%m%dT%H%M%S%fZ"),
        "algorithm": selected_name,
        "dataset_name": dataset_name,
        "dataset_sha256": checksum,
        "trained_at": now.isoformat(),
        "random_seed": seed,
        "raw_feature_names": RAW_FEATURES,
        "decision_threshold": threshold["threshold"],
        "risk_bands": {"low_max_exclusive": 30, "medium_max_exclusive": 70},
        "validation_average_precision": validation["average_precision"],
        "test_average_precision": test_metrics["average_precision"],
        "threshold_selection": threshold,
        "preprocessing_version": "1.0",
        "large_amount_cutoff": pipeline.named_steps["features"].large_amount_cutoff_,
        "ratio_cap": 1_000_000,
        "python_version": platform.python_version(),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
        "split": split,
    }
    validated = save_artifact(pipeline, metadata, output)
    test_metrics.update(model_version=validated.model_version, dataset_name=dataset_name)
    write_json(reports / "test_metrics.json", test_metrics)
    save_evaluation_plots(parts["test"].isFraud, test_scores, threshold["threshold"], reports)
    if tracking_uri:
        log_mlflow(tracking_uri, fitted, split, validated.model_dump(), output, reports)
    return test_metrics


def log_mlflow(uri: str, fitted: dict, split: dict, metadata: dict, output: Path, reports: Path):
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("TransactionGuard AI")
    for name, (pipeline, threshold, validation) in fitted.items():
        with mlflow.start_run(run_name=f"{name}-{metadata['model_version']}"):
            mlflow.log_params(
                {
                    "algorithm": name,
                    "seed": metadata["random_seed"],
                    "dataset_sha256": metadata["dataset_sha256"],
                    "raw_features": json.dumps(RAW_FEATURES),
                    "preprocessing_version": "1.0",
                    "threshold_policy": threshold["policy"],
                    **pipeline.named_steps["classifier"].get_params(),
                }
            )
            mlflow.log_metrics({f"validation_{k}": v for k, v in validation.items()})
            mlflow.log_dict(split, "split_summary.json")
            if name == metadata["algorithm"]:
                mlflow.set_tag("selected_model", "true")
                final = json.loads((reports / "test_metrics.json").read_text())
                mlflow.log_metrics(
                    {
                        f"final_test_{k}": final[k]
                        for k in ("precision", "recall", "f1", "average_precision")
                    }
                )
                mlflow.log_artifacts(str(output), artifact_path="trusted_local_model")
                mlflow.log_artifact(str(reports / "test_metrics.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--config", type=Path, default=Path("configs/training.yaml"))
    parser.add_argument("--dataset-name", default="PaySim synthetic")
    parser.add_argument(
        "--mlflow-uri", help="Optional local tracking URI, e.g. sqlite:///mlflow.db"
    )
    args = parser.parse_args()
    try:
        config = yaml.safe_load(args.config.read_text())
        print(
            json.dumps(
                train(
                    args.data, args.output, args.reports, config, args.dataset_name, args.mlflow_uri
                ),
                indent=2,
            )
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
