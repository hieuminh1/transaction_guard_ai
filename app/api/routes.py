from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from app.schemas.transaction import BatchRequest, BatchResponse, Prediction, Transaction
from app.services.predictor import Predictor

router = APIRouter()


def get_predictor(request: Request) -> Predictor:
    return request.app.state.predictor


PredictorDependency = Annotated[Predictor, Depends(get_predictor)]


def record(
    request: Request, transactions: list[Transaction], predictions: list[Prediction]
) -> None:
    request.app.state.monitor.predictions(predictions)
    history = request.app.state.history
    if history is not None and not history.record(transactions, predictions):
        request.app.state.monitor.logging_failure()


@router.get("/health")
def health(predictor: PredictorDependency):
    if not predictor.loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "model_loaded": False, "detail": predictor.error},
        )
    return {"status": "ok", "model_loaded": True, "model_version": predictor.metadata.model_version}


@router.get("/model-info")
def model_info(predictor: PredictorDependency) -> dict:
    return predictor.model_info()


@router.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request, predictor: PredictorDependency):
    predictions = predictor.predict([transaction])
    record(request, [transaction], predictions)
    return predictions[0]


@router.post("/batch-predict", response_model=BatchResponse)
def batch_predict(batch: BatchRequest, request: Request, predictor: PredictorDependency):
    if len(batch.transactions) > request.app.state.settings.max_batch_size:
        raise HTTPException(422, "Batch exceeds configured MAX_BATCH_SIZE")
    predictions = predictor.predict(batch.transactions)
    record(request, batch.transactions, predictions)
    return BatchResponse(count=len(predictions), predictions=predictions)


@router.get("/predictions")
def predictions(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
    flagged: bool | None = None,
):
    history = request.app.state.history
    if history is None:
        raise HTTPException(503, "Prediction history is not configured")
    try:
        return history.list(limit, offset, flagged)
    except Exception as exc:
        raise HTTPException(503, "Prediction history is unavailable") from exc


@router.get("/predictions/summary")
def prediction_summary(request: Request):
    history = request.app.state.history
    if history is None:
        raise HTTPException(503, "Prediction history is not configured")
    try:
        return history.summary()
    except Exception as exc:
        raise HTTPException(503, "Prediction history is unavailable") from exc


@router.get("/metrics")
def metrics(request: Request) -> dict:
    predictor = request.app.state.predictor
    return {
        **request.app.state.monitor.snapshot(),
        "model_version": predictor.metadata.model_version if predictor.loaded else None,
    }
