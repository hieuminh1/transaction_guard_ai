import json

import pytest

from app.schemas.transaction import Transaction
from app.services.predictor import Predictor
from ml.evaluate import evaluate
from ml.train import train


def test_train_reload_evaluate_exact_test(frame, tmp_path, transaction):
    data = tmp_path / "data.csv"
    frame.to_csv(data, index=False)
    output, reports = tmp_path / "artifacts", tmp_path / "reports"
    config = {
        "seed": 42,
        "minimum_precision": 0.2,
        "logistic_regression": {"max_iter": 1000},
        "random_forest": {"n_estimators": 8, "max_depth": 3, "n_jobs": 1},
    }
    final = train(data, output, reports, config, dataset_name="Artificial integration fixture")
    predictor = Predictor(output / "model.joblib", output / "metadata.json")
    assert (
        predictor.predict([Transaction(**transaction)])[0].model_version == final["model_version"]
    )
    reproduced = evaluate(
        output / "model.joblib", data, reports / "split_summary.json", reports / "reproduced"
    )
    for metric in ("precision", "recall", "f1", "average_precision", "tp", "tn", "fp", "fn"):
        assert reproduced[metric] == final[metric]
    assert (reports / "confusion_matrix.png").is_file()
    assert (reports / "precision_recall_curve.png").is_file()
    assert json.loads((reports / "split_summary.json").read_text())["dataset_sha256"]
    frame.loc[0, "amount"] += 1
    frame.to_csv(data, index=False)
    with pytest.raises(ValueError, match="checksum"):
        evaluate(output / "model.joblib", data, reports / "split_summary.json", reports)
