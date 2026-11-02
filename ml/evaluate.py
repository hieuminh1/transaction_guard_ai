"""Reproduce frozen-model test metrics on exactly the saved test partition."""

import argparse
import json
from pathlib import Path

from ml.artifact import load_artifact
from ml.data import RAW_FEATURES, file_sha256, load_dataset, reconstruct_test, write_json
from ml.metrics import classification_metrics, save_evaluation_plots


def evaluate(artifact: Path, data: Path, split_path: Path, output: Path) -> dict:
    pipeline, metadata = load_artifact(artifact, artifact.with_name("metadata.json"))
    split = json.loads(split_path.read_text())
    if split != metadata.model_extra["split"]:
        raise ValueError("Split metadata differs from the frozen model split")
    checksum = file_sha256(data)
    if checksum != metadata.dataset_sha256:
        raise ValueError("Dataset differs from model training checksum")
    test = reconstruct_test(load_dataset(data), split, checksum)
    scores = pipeline.predict_proba(test[RAW_FEATURES])[:, 1]
    metrics = {
        "algorithm": metadata.algorithm,
        "model_version": metadata.model_version,
        "split": "final_test_reproduction",
        "dataset_name": metadata.dataset_name,
        **classification_metrics(test.isFraud, scores, metadata.decision_threshold),
    }
    write_json(output / "test_metrics_reproduced.json", metrics)
    save_evaluation_plots(test.isFraud, scores, metadata.decision_threshold, output)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/reproduced"))
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate(args.artifact, args.data, args.split, args.output), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
