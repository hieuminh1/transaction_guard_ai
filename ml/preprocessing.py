from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.features import NUMERIC_FEATURES, TransactionFeatures


def build_pipeline(classifier, scale: bool = True) -> Pipeline:
    numeric = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric.append(("scaler", StandardScaler()))
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["type"]),
        ]
    )
    return Pipeline(
        [
            ("features", TransactionFeatures()),
            ("preprocessing", preprocessing),
            ("classifier", classifier),
        ]
    )
