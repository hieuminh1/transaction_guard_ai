from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    __table_args__ = (
        CheckConstraint("amount >= 0", name="amount_nonnegative"),
        CheckConstraint("fraud_probability BETWEEN 0 AND 1", name="probability_range"),
        CheckConstraint("risk_score BETWEEN 0 AND 100", name="score_range"),
        CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="risk_level_values"),
        Index("idx_prediction_logs_predicted_at", "predicted_at"),
        Index("idx_prediction_logs_flagged", "is_flagged", "predicted_at"),
    )
    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(40))
    transaction_type: Mapped[str] = mapped_column(String(16))
    amount: Mapped[float] = mapped_column(Float)
    fraud_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(SmallInteger)
    risk_level: Mapped[str] = mapped_column(String(10))
    is_flagged: Mapped[bool] = mapped_column(Boolean)
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
