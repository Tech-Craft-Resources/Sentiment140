"""
Model loading, MLflow artifact management, and inference logic.

Flow:
1. Check if best_model.pkl and best_model_card.json exist in /models
2. If not, download them from MLflow using MLFLOW_BEST_RUN_ID
3. Load model + vectorizer and cache for subsequent requests
"""

import json
import logging
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

import emoji
import joblib
import nltk
import spacy
from nltk.corpus import stopwords

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_ENCODED = BASE_DIR / "data" / "encoded"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

MODEL_PKL = MODELS_DIR / "best_model.pkl"
MODEL_CARD_JSON = MODELS_DIR / "best_model_card.json"

# ---------------------------------------------------------------------------
# NLTK setup
# ---------------------------------------------------------------------------
for resource in ["stopwords", "wordnet"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

_STOP_WORDS: set[str] = set(stopwords.words("english"))
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


# ---------------------------------------------------------------------------
# MLflow artifact download
# ---------------------------------------------------------------------------


def _download_artifacts_from_mlflow() -> None:
    """Download best_model.pkl and model_card.json from MLflow artifacts."""
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    run_id = os.getenv("MLFLOW_BEST_RUN_ID")

    if not run_id:
        raise RuntimeError(
            "MLFLOW_BEST_RUN_ID is not set. Cannot download model artifacts."
        )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Download sklearn model ---
    if not MODEL_PKL.exists():
        logger.info("Downloading sklearn model from MLflow run %s ...", run_id)
        sklearn_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        joblib.dump(sklearn_model, MODEL_PKL)
        logger.info("Model saved to %s", MODEL_PKL)

    # --- Download model_card.json ---
    if not MODEL_CARD_JSON.exists():
        logger.info("Downloading model_card.json from MLflow run %s ...", run_id)
        artifact_path = client.download_artifacts(run_id, "model_card.json")
        shutil.copy2(artifact_path, MODEL_CARD_JSON)
        logger.info("Model card saved to %s", MODEL_CARD_JSON)


def ensure_model_artifacts() -> None:
    """
    Ensure that model artifacts exist locally.
    If they don't, download them from MLflow.
    """
    if MODEL_PKL.exists() and MODEL_CARD_JSON.exists():
        logger.info("Model artifacts found locally. Skipping download.")
        return

    logger.info("Model artifacts not found locally. Downloading from MLflow...")
    _download_artifacts_from_mlflow()

    # Final validation
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Failed to obtain model at {MODEL_PKL}")
    if not MODEL_CARD_JSON.exists():
        raise FileNotFoundError(f"Failed to obtain model card at {MODEL_CARD_JSON}")


# ---------------------------------------------------------------------------
# Text preprocessing (mirrors notebook 01)
# ---------------------------------------------------------------------------


def _normalize_elongations(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    lemmatize: bool = True,
    emoji_mode: str = "drop",
    normalize_elongations_flag: bool = True,
    remove_punct: bool = True,
) -> str:
    """Mirrors the preprocessing function in 01_preprocessing.ipynb."""
    nlp = _get_nlp()
    text = str(text).lower()

    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)

    if emoji_mode == "translate":
        text = emoji.demojize(text, delimiters=(" ", " "))
    elif emoji_mode == "drop":
        text = emoji.replace_emoji(text, replace="")

    if normalize_elongations_flag:
        text = _normalize_elongations(text)

    doc = nlp(text)
    tokens = []
    for token in doc:
        if remove_punct and (token.is_punct or token.is_space):
            continue
        word = token.lemma_ if lemmatize else token.text
        word = word.strip()
        if not word:
            continue
        if remove_stopwords and word in _STOP_WORDS:
            continue
        tokens.append(word)

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_model():
    """Load best_model.pkl, model card, and vectorizer. Cached after first call."""
    clf = joblib.load(MODEL_PKL)

    with open(MODEL_CARD_JSON) as f:
        card = json.load(f)

    encoding = card.get("encoding", "tfidf_bi")
    vectorizer_path = DATA_ENCODED / f"{encoding}_vectorizer.pkl"

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {vectorizer_path}. "
            "Ensure data/encoded/ contains the required vectorizer file."
        )

    vectorizer = joblib.load(vectorizer_path)
    return clf, vectorizer, card


def get_model_card() -> dict:
    """Return the model card as a dictionary."""
    with open(MODEL_CARD_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _get_confidence(clf, X, pred: int) -> float:
    """
    Extract a confidence score from the classifier.

    Tries predict_proba first (LogisticRegression, RF, MLP).
    Falls back to decision_function + sigmoid (LinearSVC).
    """
    import numpy as np

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        return float(proba[pred])

    if hasattr(clf, "decision_function"):
        decision = clf.decision_function(X)[0]
        # Sigmoid to normalize decision_function output to [0, 1]
        prob_positive = 1.0 / (1.0 + np.exp(-float(decision)))
        return float(prob_positive if pred == 1 else 1.0 - prob_positive)

    return 0.0


def predict(text: str) -> dict:
    """Run inference on a single raw text."""
    preproc_config_path = DATA_PROCESSED / "best_preproc_config.json"

    if not preproc_config_path.exists():
        raise FileNotFoundError(
            f"Preprocessing config not found at {preproc_config_path}. "
            "Ensure data/processed/best_preproc_config.json exists."
        )

    with open(preproc_config_path) as f:
        preproc_config = json.load(f)

    clf, vectorizer, card = load_model()

    cleaned = preprocess_text(text, **preproc_config)
    X = vectorizer.transform([cleaned])
    pred = clf.predict(X)[0]
    confidence = _get_confidence(clf, X, int(pred))

    label_str = "POSITIVE" if int(pred) == 1 else "NEGATIVE"
    return {
        "text": text,
        "label": label_str,
        "label_id": int(pred),
        "confidence": round(confidence, 4),
    }
