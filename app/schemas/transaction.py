from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TransactionType = Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


class Transaction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "transaction_id": "txn-1001",
                    "step": 42,
                    "type": "TRANSFER",
                    "amount": 12000.0,
                    "oldbalanceOrg": 15000.0,
                }
            ]
        },
    )
    transaction_id: str = Field(min_length=1, max_length=64)
    step: int = Field(ge=0, strict=True)
    type: TransactionType
    amount: float = Field(ge=0, strict=True)
    oldbalanceOrg: float = Field(ge=0, strict=True)

    @field_validator("transaction_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transaction_id cannot be blank")
        return value


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.transaction_id for item in self.transactions]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate transaction_id within batch")
        return self


class Prediction(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1, allow_inf_nan=False)
    risk_score: int = Field(ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]
    is_flagged: bool
    decision_threshold: float = Field(ge=0, le=1)
    model_version: str


class BatchResponse(BaseModel):
    count: int
    predictions: list[Prediction]
