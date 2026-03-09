"""
FastAPI inference API for the Sentiment140 project.

Endpoints:
    GET  /health      — Health check
    POST /predict     — Predict sentiment for a single text
    GET  /model_card  — Return the model card JSON

Startup:
    1. Validates MLFLOW_BEST_RUN_ID is set (fails fast if missing)
    2. Ensures model artifacts exist locally (downloads from MLflow if needed)
    3. Pre-loads the model into memory
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from inference import ensure_model_artifacts, get_model_card, load_model, predict
from schemas import PredictRequest, PredictResponse, PredictionItem

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup validation + model preload
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown logic for the application."""

    # --- Validate required env vars ---
    run_id = os.getenv("MLFLOW_BEST_RUN_ID")
    if not run_id:
        raise RuntimeError(
            "MLFLOW_BEST_RUN_ID environment variable is not set. "
            "The API cannot start without a valid MLflow run ID. "
            "Set it in the .env file or as an environment variable."
        )

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI environment variable is not set. "
            "Set it in the .env file or as an environment variable."
        )

    logger.info("MLFLOW_BEST_RUN_ID = %s", run_id)
    logger.info("MLFLOW_TRACKING_URI = %s", tracking_uri)

    # --- Ensure artifacts exist (download from MLflow if needed) ---
    try:
        ensure_model_artifacts()
    except Exception as e:
        raise RuntimeError(f"Failed to ensure model artifacts: {e}") from e

    # --- Pre-load model into memory ---
    try:
        load_model()
        logger.info("Model pre-loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to pre-load model: {e}") from e

    yield  # Application is running

    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Sentiment140 Inference API",
    description=(
        "Sentiment analysis API powered by a classical ML model trained on "
        "the Sentiment140 dataset. The model is managed via MLflow and "
        "automatically downloaded on first startup."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Utility"])
def health():
    """Health check."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_endpoint(request: PredictRequest):
    """
    Predict sentiment for a single text.

    Returns POSITIVE or NEGATIVE label along with model metadata.
    """
    try:
        result = predict(request.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    _, _, card = load_model()
    item = PredictionItem(**result)

    return PredictResponse(
        prediction=item,
        model=card.get("model_name", "Unknown"),
        encoding=card.get("encoding", "Unknown"),
    )


@app.get("/model_card", tags=["Model"])
def model_card():
    """Return the full model card JSON."""
    try:
        card = get_model_card()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model card not available.",
        )
    return card
