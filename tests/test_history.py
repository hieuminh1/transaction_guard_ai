import pytest

pytest.importorskip("sqlalchemy")

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.db.models import Base
from app.db.session import History
from app.main import create_app


def test_history_persistence_filtering_and_repeated_ids(artifact, tmp_path, transaction):
    url = f"sqlite:///{tmp_path / 'history.db'}"
    history = History(url)
    Base.metadata.create_all(history.engine)
    history.close()
    with TestClient(
        create_app(
            Settings(
                _env_file=None,
                database_url=url,
                model_path=artifact / "model.joblib",
                model_metadata_path=artifact / "metadata.json",
            )
        )
    ) as c:
        for _ in range(2):
            scored = c.post("/predict", json=transaction)
            assert scored.status_code == 200
        result = c.get("/predictions?limit=1&offset=1").json()
        assert result["total"] == 2
        assert len(result["items"]) == 1
        assert result["items"][0]["transaction_id"] == transaction["transaction_id"]
        flag = str(scored.json()["is_flagged"]).lower()
        assert c.get(f"/predictions?flagged={flag}").json()["total"] == 2
        summary = c.get("/predictions/summary").json()
        assert summary["total_scored"] == 2
        assert summary["average_model_score"] == scored.json()["risk_score"]
