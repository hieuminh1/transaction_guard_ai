import logging

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.db.models import PredictionLog

logger = logging.getLogger(__name__)


class History:
    def __init__(self, url: str):
        options = {"connect_timeout": 2} if url.startswith("postgresql") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=options)

    def close(self) -> None:
        self.engine.dispose()

    def record(self, transactions: list, predictions: list) -> bool:
        try:
            with Session(self.engine) as session, session.begin():
                session.add_all(
                    [
                        PredictionLog(
                            transaction_id=transaction.transaction_id,
                            transaction_type=transaction.type,
                            amount=transaction.amount,
                            **prediction.model_dump(
                                exclude={"transaction_id", "decision_threshold"}
                            ),
                        )
                        for transaction, prediction in zip(transactions, predictions, strict=True)
                    ]
                )
            return True
        except Exception:
            # Do not log SQL errors/parameters, transaction data, or connection credentials.
            logger.warning("Prediction history write failed; inference response preserved")
            return False

    def list(self, limit: int, offset: int, flagged: bool | None) -> dict:
        with Session(self.engine) as session:
            query = select(PredictionLog)
            count = select(func.count()).select_from(PredictionLog)
            if flagged is not None:
                query = query.where(PredictionLog.is_flagged == flagged)
                count = count.where(PredictionLog.is_flagged == flagged)
            rows = session.scalars(
                query.order_by(PredictionLog.predicted_at.desc(), PredictionLog.id.desc())
                .limit(limit)
                .offset(offset)
            ).all()
            return {
                "items": [
                    {
                        column.name: getattr(row, column.name)
                        for column in PredictionLog.__table__.columns
                    }
                    for row in rows
                ],
                "total": session.scalar(count),
                "limit": limit,
                "offset": offset,
            }

    def summary(self) -> dict:
        with Session(self.engine) as session:
            total = session.scalar(select(func.count()).select_from(PredictionLog))
            flagged = session.scalar(
                select(func.count())
                .select_from(PredictionLog)
                .where(PredictionLog.is_flagged.is_(True))
            )
            average = session.scalar(select(func.avg(PredictionLog.risk_score)))
            version = session.scalar(
                select(PredictionLog.model_version)
                .order_by(PredictionLog.predicted_at.desc(), PredictionLog.id.desc())
                .limit(1)
            )
            return {
                "total_scored": total,
                "flagged_count": flagged,
                "flagged_rate": flagged / total if total else 0,
                "average_model_score": average,
                "latest_model_version": version,
            }
