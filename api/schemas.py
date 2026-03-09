"""Pydantic schemas for the Sentiment140 inference API."""

from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    """Single text input for sentiment prediction."""

    text: str = Field(
        ...,
        min_length=1,
        description="A tweet or text to classify",
    )

    model_config = {
        "json_schema_extra": {"examples": [{"text": "I love this so much!"}]}
    }


class PredictionItem(BaseModel):
    """Result of a single sentiment prediction."""

    text: str
    label: Literal["POSITIVE", "NEGATIVE"]
    label_id: int
    confidence: float | None = None


class PredictResponse(BaseModel):
    """Response wrapping a single prediction plus model metadata."""

    prediction: PredictionItem
    model: str
    encoding: str
