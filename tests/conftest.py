from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.core.config import Settings
from app.main import create_app
from ml.artifact import save_artifact
from ml.data import RAW_FEATURES
from ml.demo_data import generate_demo_data
from ml.preprocessing import build_pipeline


@pytest.fixture
def frame():
    return generate_demo_data(steps=20, rows_per_step=10)


@pytest.fixture
def transaction():
    return {
        "transaction_id": "test-1",
        "step": 42,
        "type": "TRANSFER",
        "amount": 12000.0,
        "oldbalanceOrg": 15000.0,
    }


@pytest.fixture
def artifact(tmp_path: Path, frame):
    pipeline = build_pipeline(LogisticRegression(random_state=42, max_iter=1000))
    pipeline.fit(frame[RAW_FEATURES], frame.isFraud)
    save_artifact(
        pipeline,
        {
            "schema_version": "1.0",
            "model_version": "test-v1",
            "algorithm": "LogisticRegression",
            "dataset_name": "Artificial test fixture",
            "dataset_sha256": "a" * 64,
            "trained_at": "2026-01-01T00:00:00+00:00",
            "random_seed": 42,
            "raw_feature_names": RAW_FEATURES,
            "decision_threshold": 0.2,
            "risk_bands": {"low_max_exclusive": 30, "medium_max_exclusive": 70},
            "validation_average_precision": 0.1,
            "test_average_precision": 0.1,
        },
        tmp_path,
    )
    return tmp_path


@pytest.fixture
def client(artifact):
    settings = Settings(
        _env_file=None,
        database_url=None,
        model_path=artifact / "model.joblib",
        model_metadata_path=artifact / "metadata.json",
    )
    with TestClient(create_app(settings)) as client:
        yield client
