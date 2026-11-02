import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from ml.data import RAW_FEATURES
from ml.features import RATIO_CAP, TransactionFeatures
from ml.preprocessing import build_pipeline


def test_safe_extremes_and_no_leakage(frame):
    transformer = TransactionFeatures().fit(frame)
    cutoff = transformer.large_amount_cutoff_
    data = frame.iloc[:3].copy()
    data["oldbalanceOrg"] = 0.0
    data["amount"] = [0, 1e100, np.finfo(float).max]
    output = transformer.transform(data)
    assert np.isfinite(output.drop(columns="type").to_numpy()).all()
    assert output.amount_to_balance_ratio.max() <= RATIO_CAP
    assert "isFraud" not in output
    assert transformer.large_amount_cutoff_ == cutoff
    assert cutoff == frame.amount.quantile(0.95)


def test_pipeline_round_trip(frame, tmp_path):
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(frame, frame.isFraud)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(pipeline.predict_proba(frame), restored.predict_proba(frame))
    changed = frame.copy()
    changed["isFraud"] = 1 - changed.isFraud
    np.testing.assert_allclose(pipeline.predict_proba(frame), pipeline.predict_proba(changed))
    features = restored[:-1].transform(frame[RAW_FEATURES])
    np.testing.assert_allclose(features, pipeline[:-1].transform(frame))


def test_unknown_type_rejected(frame):
    transformer = TransactionFeatures().fit(frame)
    frame.loc[0, "type"] = "UNKNOWN"
    with pytest.raises(ValueError, match="type"):
        transformer.transform(frame)
