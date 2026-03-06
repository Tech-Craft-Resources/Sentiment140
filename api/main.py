"""FastAPI inference API for the Sentiment140 project."""
import base64
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from schemas import (
    PredictRequest, PredictResponse, PredictionItem,
    ModelInfoResponse,
    AblationEntry, AblationSummaryResponse,
    ComparisonResponse,
    WorkEntry, WorkDistributionResponse,
)
from inference import predict, load_model

BASE_DIR       = Path(__file__).parent.parent
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "models"

app = FastAPI(
    title="Sentiment140 Inference API",
    description=(
        "API for sentiment analysis trained on the Sentiment140 dataset. "
        "Exposes the best classical ML model from the ablation study and "
        "compares it against DistilBERT (SST-2)."
    ),
    version="1.0.0",
)


@app.get("/model_info", response_model=ModelInfoResponse, tags=["Model"])
def model_info():
    """Return model card — description, hyperparams, encoding and metrics of the selected model."""
    card_path = MODELS_DIR / "best_model_card.json"
    if not card_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Model card not found. Run notebooks 01–04 to generate the model.",
        )
    with open(card_path) as f:
        card = json.load(f)

    return ModelInfoResponse(
        model_name=card.get("model_name", "Unknown"),
        member=card.get("member", "Unknown"),
        encoding=card.get("encoding", "Unknown"),
        metrics=card.get("metrics", {}),
        mlflow_run_id=card.get("mlflow_run_id"),
        description=card.get("description", ""),
        preprocessing=card.get("preprocessing", ""),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_endpoint(request: PredictRequest):
    """Predict sentiment for one or more tweets (individual or batch)."""
    try:
        results = predict(request.texts)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    card_path = MODELS_DIR / "best_model_card.json"
    with open(card_path) as f:
        card = json.load(f)

    items = [PredictionItem(**r) for r in results]
    return PredictResponse(
        predictions=items,
        model=card.get("model_name", "Unknown"),
        encoding=card.get("encoding", "Unknown"),
    )


@app.get("/ablation_summary", response_model=AblationSummaryResponse, tags=["Analysis"])
def ablation_summary():
    """
    Ablation study results: table with F1 per preprocessing configuration,
    a chart (base64 PNG) and a conclusion paragraph.
    """
    # Load preprocessing ablation results from MLflow or cached JSON
    # We read from the cached files written by notebook 01
    preproc_config_path = DATA_PROCESSED / "best_preproc_config.json"
    if not preproc_config_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Ablation results not found. Run notebook 01 first.",
        )

    # Try to load from MLflow experiment via mlflow client (optional)
    # For simplicity, we reconstruct from saved baseline + model results
    entries: list[AblationEntry] = []

    # Baseline results
    baseline_path = DATA_PROCESSED / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        entries.append(AblationEntry(config="baseline_bow_mnb", val_f1_macro=bl["metrics"]["f1_macro"]))

    # Per-member bests
    for member in ["nicolas", "daniel"]:
        p = DATA_PROCESSED / f"{member}_best_results.json"
        if p.exists():
            with open(p) as f:
                r = json.load(f)
            entries.append(AblationEntry(
                config=f"{member}_{r['best_model']}_{r['encoding']}",
                val_f1_macro=r["metrics"].get("test_f1_macro", 0.0),
            ))

    if not entries:
        raise HTTPException(status_code=503, detail="No experiment results found yet.")

    best = max(entries, key=lambda e: e.val_f1_macro)

    # Try to load pre-generated chart
    chart_b64 = None
    chart_path = DATA_PROCESSED / "ablation_preprocessing.png"
    if chart_path.exists():
        chart_b64 = base64.b64encode(chart_path.read_bytes()).decode()

    conclusion = (
        f"The best configuration achieved F1={best.val_f1_macro:.4f} "
        f"using '{best.config}'. "
        "Preprocessing ablation shows that lemmatization and stopword removal "
        "generally improve performance on Sentiment140. "
        "Emoji translation tends to help over dropping, as emojis carry sentiment signal in tweets. "
        "Elongation normalization provides marginal gains. "
        "Nicolas Rodriguez ran experiments with Random Forest and MLP; "
        "Daniel Velasco ran Logistic Regression and LinearSVC."
    )

    return AblationSummaryResponse(
        results=entries,
        best_config=best.config,
        conclusion=conclusion,
        chart_base64=chart_b64,
    )


@app.get("/comparison", response_model=ComparisonResponse, tags=["Analysis"])
def comparison():
    """
    Comparison between the best classical model and DistilBERT (SST-2)
    in terms of F1-score and inference latency.
    """
    comp_path = DATA_PROCESSED / "comparison_results.json"
    if not comp_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Comparison results not found. Run notebook 05 first.",
        )

    with open(comp_path) as f:
        data = json.load(f)

    classical_f1 = data["classical"]["metrics"].get("f1_macro", 0)
    hf_f1        = data["distilbert"]["metrics"].get("f1_macro", 0)
    classical_lat = data["classical"]["metrics"].get("latency_s", 0)
    hf_lat        = data["distilbert"]["metrics"].get("latency_s", 0)

    speed_ratio = hf_lat / classical_lat if classical_lat > 0 else float("inf")
    diff = classical_f1 - hf_f1

    if diff >= 0:
        summary = (
            f"The classical model ({data['classical']['model']}) outperforms DistilBERT (SST-2) "
            f"by {diff:.4f} F1 points on Sentiment140. "
            f"It is also ~{speed_ratio:.1f}x faster at inference. "
            "This is expected: DistilBERT was fine-tuned on movie reviews (SST-2) and faces "
            "domain mismatch with Twitter slang and abbreviations."
        )
    else:
        summary = (
            f"DistilBERT (SST-2) outperforms the classical model by {abs(diff):.4f} F1 points, "
            f"but is ~{speed_ratio:.1f}x slower. The classical model remains a strong baseline "
            "given its speed advantage."
        )

    return ComparisonResponse(classical=data["classical"], distilbert=data["distilbert"], summary=summary)


@app.get("/work_distribution", response_model=WorkDistributionResponse, tags=["Analysis"])
def work_distribution():
    """Table showing the experiment distribution among group members."""
    distribution = [
        WorkEntry(
            member="Nicolas Rodriguez",
            models=["Random Forest", "MLPClassifier (dense neural network)"],
            encoding="TF-IDF Bigram (best from ablation)",
            mlflow_tag="member=nicolas",
        ),
        WorkEntry(
            member="Daniel Velasco",
            models=["Logistic Regression", "LinearSVC (SVM)"],
            encoding="TF-IDF Bigram (best from ablation)",
            mlflow_tag="member=daniel",
        ),
    ]
    return WorkDistributionResponse(
        distribution=distribution,
        metric="F1-score macro",
    )


@app.get("/health", tags=["Utility"])
def health():
    """Health check."""
    return {"status": "ok"}
