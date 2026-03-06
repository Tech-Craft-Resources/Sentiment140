from pydantic import BaseModel, Field
from typing import Literal


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="One or more tweet texts to classify")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["I love this so much!", "This is terrible and disappointing."]}
            ]
        }
    }


class PredictionItem(BaseModel):
    text: str
    label: Literal["POSITIVE", "NEGATIVE"]
    label_id: int
    confidence: float | None = None


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    model: str
    encoding: str


class ModelInfoResponse(BaseModel):
    model_name: str
    member: str
    encoding: str
    metrics: dict[str, float]
    mlflow_run_id: str | None = None
    description: str
    preprocessing: str


class AblationEntry(BaseModel):
    config: str
    val_f1_macro: float


class AblationSummaryResponse(BaseModel):
    results: list[AblationEntry]
    best_config: str
    conclusion: str
    chart_base64: str | None = None


class ComparisonResponse(BaseModel):
    classical: dict
    distilbert: dict
    summary: str


class WorkEntry(BaseModel):
    member: str
    models: list[str]
    encoding: str
    mlflow_tag: str


class WorkDistributionResponse(BaseModel):
    distribution: list[WorkEntry]
    metric: str
