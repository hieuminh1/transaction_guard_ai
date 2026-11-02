import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_predict_schema_and_repeat(client, transaction):
    first = client.post("/predict", json=transaction)
    assert first.status_code == 200
    result = first.json()
    assert result == client.post("/predict", json=transaction).json()
    assert 0 <= result["fraud_probability"] <= 1
    assert result["risk_score"] == round(100 * result["fraud_probability"])
    assert result["is_flagged"] == (result["fraud_probability"] >= 0.2)
    assert "isFraud" not in result
    assert client.get("/health").status_code == 200
    info = client.get("/model-info").json()
    assert info["model_version"] == "test-v1"
    assert info["decision_threshold"] == 0.2


@pytest.mark.parametrize(
    "changes",
    [
        {"amount": -1},
        {"oldbalanceOrg": -1},
        {"step": -1},
        {"step": 1.5},
        {"step": True},
        {"amount": True},
        {"amount": "12"},
        {"isFraud": 1},
        {"type": "OTHER"},
        {"transaction_id": " "},
        {"transaction_id": "a" * 65},
    ],
)
def test_bad_input_rejected(client, transaction, changes):
    assert client.post("/predict", json={**transaction, **changes}).status_code == 422


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_returns_json_validation_error(client, transaction, value):
    transaction["amount"] = value
    response = client.post(
        "/predict", content=json.dumps(transaction), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_missing_input(client, transaction):
    del transaction["amount"]
    assert client.post("/predict", json=transaction).status_code == 422


@pytest.mark.parametrize("amount,balance", [(0, 0), (1e300, 0)])
def test_extreme_valid_inputs(client, transaction, amount, balance):
    transaction.update(amount=amount, oldbalanceOrg=balance)
    assert client.post("/predict", json=transaction).status_code == 200


def test_batch_validation_and_vectorization(client, transaction):
    assert client.post("/batch-predict", json={"transactions": []}).status_code == 422
    assert (
        client.post("/batch-predict", json={"transactions": [transaction] * 2}).status_code == 422
    )
    batch = [{**transaction, "transaction_id": f"t-{i}"} for i in range(101)]
    assert client.post("/batch-predict", json={"transactions": batch}).status_code == 422
    pipeline = client.app.state.predictor.pipeline
    with patch.object(pipeline, "predict_proba", wraps=pipeline.predict_proba) as predict:
        response = client.post("/batch-predict", json={"transactions": batch[:100]})
        assert response.status_code == 200
        assert predict.call_count == 1
    assert response.json()["count"] == 100
    assert [p["transaction_id"] for p in response.json()["predictions"]] == [
        item["transaction_id"] for item in batch[:100]
    ]


def test_missing_model_safe(tmp_path, transaction):
    with TestClient(
        create_app(
            Settings(
                _env_file=None,
                database_url=None,
                model_path=tmp_path / "missing.joblib",
                model_metadata_path=tmp_path / "missing.json",
            )
        )
    ) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json=transaction).status_code == 503
        assert c.get("/model-info").status_code == 503
        assert str(tmp_path) not in c.get("/health").text


def test_operational_metrics_and_history_disabled(client, transaction):
    client.post("/predict", json=transaction)
    metrics = client.get("/metrics").json()
    assert metrics["total_scored"] == 1
    assert metrics["request_count"] >= 1
    assert client.get("/predictions").status_code == 503
    assert client.get("/predictions?limit=101").status_code == 422
    assert client.get("/missing").status_code == 404


def test_internal_errors_safe(client, transaction):
    with patch.object(client.app.state.predictor, "predict", side_effect=RuntimeError("private")):
        response = client.post("/predict", json=transaction)
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error"}


def test_database_outage_is_fail_open(client, transaction):
    from unittest.mock import Mock

    history = Mock()
    history.record.return_value = False
    client.app.state.history = history
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    assert client.get("/metrics").json()["database_log_failures"] == 1
