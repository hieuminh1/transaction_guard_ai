import json

import pytest

from app.schemas.transaction import Transaction
from app.services.predictor import ModelUnavailable, Predictor


def test_artifact_consistent(artifact, transaction):
    predictor = Predictor(artifact / "model.joblib", artifact / "metadata.json")
    assert predictor.loaded
    item = Transaction(**transaction)
    assert predictor.predict([item]) == predictor.predict([item])
    assert predictor.model_info()["decision_threshold"] == 0.2


@pytest.mark.parametrize("corruption", ["version", "threshold", "schema", "checksum", "binary"])
def test_corrupt_artifact_unavailable(artifact, corruption, transaction):
    path = artifact / "metadata.json"
    metadata = json.loads(path.read_text())
    if corruption == "version":
        metadata["model_version"] = "wrong"
    elif corruption == "threshold":
        metadata["decision_threshold"] = float("nan")
    elif corruption == "schema":
        metadata["schema_version"] = "999"
    elif corruption == "checksum":
        metadata["artifact_sha256"] = "b" * 64
    else:
        (artifact / "model.joblib").write_bytes(b"invalid")
    path.write_text(json.dumps(metadata))
    predictor = Predictor(artifact / "model.joblib", path)
    assert not predictor.loaded
    with pytest.raises(ModelUnavailable):
        predictor.predict([Transaction(**transaction)])
