import logging
from pathlib import Path

import numpy as np
import pandas as pd

from app.schemas.transaction import Prediction, Transaction
from ml.artifact import load_artifact
from ml.data import RAW_FEATURES

logger = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    pass


class Predictor:
    def __init__(self, model_path: Path, metadata_path: Path):
        self.pipeline = None
        self.metadata = None
        self.error = "Model is unavailable"
        try:
            self.pipeline, self.metadata = load_artifact(model_path, metadata_path)
        except FileNotFoundError:
            self.error = "Model artifact not found"
            logger.warning("Model artifact not found; inference is unavailable")
        except Exception:
            self.error = "Model artifact is invalid or incompatible"
            logger.exception("Trusted local model could not be loaded")

    @property
    def loaded(self) -> bool:
        return self.pipeline is not None and self.metadata is not None

    def require_loaded(self) -> None:
        if not self.loaded:
            raise ModelUnavailable(self.error)

    def predict(self, transactions: list[Transaction]) -> list[Prediction]:
        self.require_loaded()
        frame = pd.DataFrame([item.model_dump() for item in transactions], columns=RAW_FEATURES)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        if (
            not np.isfinite(probabilities).all()
            or ((probabilities < 0) | (probabilities > 1)).any()
        ):
            raise RuntimeError("Classifier returned invalid probability scores")
        results = []
        for transaction, probability in zip(transactions, probabilities, strict=True):
            score = round(float(probability) * 100)
            results.append(
                Prediction(
                    transaction_id=transaction.transaction_id,
                    fraud_probability=float(probability),
                    risk_score=score,
                    risk_level="low" if score < 30 else "medium" if score < 70 else "high",
                    is_flagged=bool(probability >= self.metadata.decision_threshold),
                    decision_threshold=self.metadata.decision_threshold,
                    model_version=self.metadata.model_version,
                )
            )
        return results

    def model_info(self) -> dict:
        self.require_loaded()
        return {
            "model_version": self.metadata.model_version,
            "algorithm": self.metadata.algorithm,
            "schema_version": self.metadata.schema_version,
            "decision_threshold": self.metadata.decision_threshold,
            "feature_names": self.metadata.raw_feature_names,
            "trained_at": self.metadata.trained_at,
            "dataset": self.metadata.dataset_name,
        }
