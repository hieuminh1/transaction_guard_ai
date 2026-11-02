"""Initial prediction history schema."""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "prediction_logs",
        sa.Column("id", sa.BigInteger().with_variant(sa.Integer(), "sqlite"), primary_key=True),
        sa.Column("transaction_id", sa.String(64), nullable=False),
        sa.Column("model_version", sa.String(40), nullable=False),
        sa.Column("transaction_type", sa.String(16), nullable=False),
        sa.Column("amount", sa.Float(), nullable=False),
        sa.Column("fraud_probability", sa.Float(), nullable=False),
        sa.Column("risk_score", sa.SmallInteger(), nullable=False),
        sa.Column("risk_level", sa.String(10), nullable=False),
        sa.Column("is_flagged", sa.Boolean(), nullable=False),
        sa.Column(
            "predicted_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("amount >= 0", name="amount_nonnegative"),
        sa.CheckConstraint("fraud_probability BETWEEN 0 AND 1", name="probability_range"),
        sa.CheckConstraint("risk_score BETWEEN 0 AND 100", name="score_range"),
        sa.CheckConstraint("risk_level IN ('low', 'medium', 'high')", name="risk_level_values"),
    )
    op.create_index("idx_prediction_logs_predicted_at", "prediction_logs", ["predicted_at"])
    op.create_index(
        "idx_prediction_logs_flagged", "prediction_logs", ["is_flagged", "predicted_at"]
    )


def downgrade() -> None:
    op.drop_table("prediction_logs")
