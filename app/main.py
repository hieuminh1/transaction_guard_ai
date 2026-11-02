import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings
from app.services.monitoring import Monitor
from app.services.predictor import ModelUnavailable, Predictor

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        logging.basicConfig(level=settings.log_level.upper())
        application.state.settings = settings
        application.state.predictor = predictor or Predictor(
            settings.model_path, settings.model_metadata_path
        )
        application.state.monitor = Monitor()
        application.state.history = None
        if settings.database_url:
            try:
                from app.db.session import History

                application.state.history = History(settings.database_url)
            except Exception:
                logger.exception("History initialization failed; prediction remains available")
        yield
        if application.state.history is not None:
            application.state.history.close()

    application = FastAPI(
        title="TransactionGuard AI",
        version="0.1.0",
        lifespan=lifespan,
        description="Synthetic demo. Scores indicate manual review, not payment rejection.",
    )
    application.include_router(router)

    @application.exception_handler(ModelUnavailable)
    async def unavailable_handler(request: Request, exc: ModelUnavailable):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @application.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        # Omit raw input/context: NaN/Infinity are not JSON serializable and inputs may be private.
        errors = [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    @application.middleware("http")
    async def observed_request(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled API error")
            response = JSONResponse(status_code=500, content={"detail": "Internal server error"})
        request.app.state.monitor.request(
            (time.perf_counter() - start) * 1000, response.status_code
        )
        return response

    return application


app = create_app()
