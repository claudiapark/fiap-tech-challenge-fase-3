"""Validated API request and response contracts."""

from typing import Literal

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Medical report text submitted for academic triage."""

    report_text: str = Field(min_length=10, max_length=5000)


class TriageResponse(BaseModel):
    """Predicted urgency with model confidence."""

    urgency: Literal["normal", "attention", "urgent"]
    confidence: float = Field(ge=0, le=1)
    engine: Literal["onnx", "sklearn"]
    disclaimer: str


class HealthResponse(BaseModel):
    """Service health contract."""

    status: Literal["ok"]
    model_loaded: bool
    engine: str
