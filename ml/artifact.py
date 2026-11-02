"""Trusted-local artifact bundle with matching version and file checksum."""

import json
from pathlib import Path
from typing import Literal

import joblib
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ml.data import RAW_FEATURES, file_sha256, write_json


class RiskBands(BaseModel):
    low_max_exclusive: Literal[30] = 30
    medium_max_exclusive: Literal[70] = 70


class ModelMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    schema_version: Literal["1.0"]
    model_version: str = Field(min_length=1, max_length=40)
    algorithm: Literal["LogisticRegression", "RandomForestClassifier"]
    dataset_name: str
    dataset_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    trained_at: str
    random_seed: int
    raw_feature_names: list[str]
    decision_threshold: float = Field(ge=0, le=1)
    risk_bands: RiskBands
    validation_average_precision: float = Field(ge=0, le=1)
    test_average_precision: float = Field(ge=0, le=1)

    @field_validator("raw_feature_names")
    @classmethod
    def validate_feature_names(cls, value: list[str]) -> list[str]:
        if value != RAW_FEATURES:
            raise ValueError("Artifact feature order differs from approved schema")
        return value


def save_artifact(pipeline, metadata: dict, output: Path) -> ModelMetadata:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "model.joblib"
    joblib.dump({"pipeline": pipeline, "model_version": metadata["model_version"]}, path)
    validated = ModelMetadata(**metadata, artifact_sha256=file_sha256(path))
    write_json(output / "metadata.json", validated.model_dump())
    return validated


def load_artifact(model_path: Path, metadata_path: Path) -> tuple:
    """Only call on trusted local files. Checksums detect mismatch, not malicious pickle."""
    metadata = ModelMetadata.model_validate(json.loads(metadata_path.read_text(encoding="utf-8")))
    if file_sha256(model_path) != metadata.artifact_sha256:
        raise ValueError("Artifact checksum mismatch")
    bundle = joblib.load(model_path)
    if bundle["model_version"] != metadata.model_version:
        raise ValueError("Model and metadata versions differ")
    pipeline = bundle["pipeline"]
    if type(pipeline.named_steps["classifier"]).__name__ != metadata.algorithm:
        raise ValueError("Classifier and metadata algorithms differ")
    if not hasattr(pipeline, "predict_proba") or list(pipeline.classes_) != [0, 1]:
        raise ValueError("Expected a fitted binary classifier with classes [0, 1]")
    return pipeline, metadata
