import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ml.data import RAW_FEATURES, validate_features

NUMERIC_FEATURES = [
    "step",
    "log_amount",
    "log_oldbalance",
    "amount_to_balance_ratio",
    "is_large_amount",
]
RATIO_CAP = 1_000_000.0


class TransactionFeatures(TransformerMixin, BaseEstimator):
    """Train-only 95th percentile; ratio capped at fixed 1e6 before division."""

    def fit(self, X: pd.DataFrame, y=None):
        validate_features(X)
        self.large_amount_cutoff_ = float(X.amount.quantile(0.95))
        self.feature_names_in_ = np.array(RAW_FEATURES, dtype=object)
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "large_amount_cutoff_")
        validate_features(X)
        frame = X.loc[:, RAW_FEATURES].copy()
        denominator = np.maximum(frame.oldbalanceOrg.to_numpy(dtype=float), 1.0)
        # Divide first: denominator >= 1, so even max finite float cannot overflow.
        ratio = np.minimum(frame.amount.to_numpy(dtype=float) / denominator, RATIO_CAP)
        return pd.DataFrame(
            {
                "step": frame.step.astype(float),
                "type": frame.type,
                "log_amount": np.log1p(frame.amount.astype(float)),
                "log_oldbalance": np.log1p(frame.oldbalanceOrg.astype(float)),
                "amount_to_balance_ratio": ratio,
                "is_large_amount": (frame.amount > self.large_amount_cutoff_).astype(float),
            },
            index=frame.index,
        )
