"""FastAPI service instrumented for Prometheus monitoring."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from medical_triage.api.schemas import HealthResponse, TriageRequest, TriageResponse
from medical_triage.config import get_settings
from medical_triage.inference import TriagePredictor

LOGGER = logging.getLogger(__name__)
DISCLAIMER = "Resultado acadêmico; não substitui avaliação de profissional de saúde."

REQUESTS = Counter(
    "triage_requests_total",
    "Total HTTP requests to the triage API",
    ("method", "path", "status"),
)
LATENCY = Histogram(
    "triage_request_latency_seconds",
    "HTTP request latency in seconds",
    ("method", "path"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
PREDICTIONS = Counter(
    "triage_predictions_total",
    "Model predictions by urgency class",
    ("urgency", "engine"),
)
ERRORS = Counter(
    "triage_errors_total",
    "Unhandled API errors",
    ("exception_type",),
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model artifacts once during application startup."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    app.state.predictor = TriagePredictor(settings.model_dir, settings.inference_engine)
    yield


app = FastAPI(
    title="Medical Text Triage API",
    version="0.1.0",
    description="Academic NLP triage service. It must not be used for clinical decisions.",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    """Record request count, status and latency without logging report text."""
    start = perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        ERRORS.labels(type(exc).__name__).inc()
        LOGGER.exception("Unhandled request error")
        raise
    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    REQUESTS.labels(request.method, path, str(response.status_code)).inc()
    LATENCY.labels(request.method, path).observe(perf_counter() - start)
    return response


@app.get("/", tags=["Service"])
def root() -> dict[str, str]:
    """Describe the service and its safety boundary."""
    return {
        "service": "medical-text-triage",
        "docs": "/docs",
        "metrics": "/metrics",
        "disclaimer": DISCLAIMER,
    }


@app.get("/health", response_model=HealthResponse, tags=["Service"])
def health(request: Request) -> HealthResponse:
    """Return readiness after successful model loading."""
    predictor = request.app.state.predictor
    return HealthResponse(status="ok", model_loaded=True, engine=predictor.engine)


@app.post("/predict", response_model=TriageResponse, tags=["Inference"])
def predict(payload: TriageRequest, request: Request) -> TriageResponse:
    """Predict urgency for one report text."""
    try:
        prediction = request.app.state.predictor.predict(payload.report_text)
    except Exception as exc:
        ERRORS.labels(type(exc).__name__).inc()
        raise HTTPException(status_code=500, detail="Model inference failed") from exc
    PREDICTIONS.labels(prediction.urgency, prediction.engine).inc()
    return TriageResponse(
        urgency=prediction.urgency,
        confidence=prediction.confidence,
        engine=prediction.engine,
        disclaimer=DISCLAIMER,
    )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
