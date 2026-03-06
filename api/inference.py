"""Model loading and inference logic."""
import re
from pathlib import Path
from functools import lru_cache

import emoji
import spacy
import nltk
import joblib
from nltk.corpus import stopwords

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"

# NLTK setup
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


@lru_cache(maxsize=1)
def load_model():
    """Load best_model.pkl and its vectorizer. Cached after first call."""
    import json
    import scipy.sparse as sp

    model_path = MODELS_DIR / "best_model.pkl"
    card_path  = MODELS_DIR / "best_model_card.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"best_model.pkl not found at {model_path}. "
            "Run notebooks 01-04 first to generate the model."
        )

    clf = joblib.load(model_path)

    with open(card_path) as f:
        card = json.load(f)

    encoding = card.get("encoding", "tfidf_bi")
    vectorizer_path = BASE_DIR / "data" / "encoded" / f"{encoding}_vectorizer.pkl"

    if not vectorizer_path.exists():
        raise FileNotFoundError(
            f"Vectorizer not found at {vectorizer_path}. Run notebook 02 first."
        )

    vectorizer = joblib.load(vectorizer_path)
    return clf, vectorizer, card


def predict(texts: list[str]) -> list[dict]:
    """Run inference on a list of raw tweet texts."""
    import json

    card_path = MODELS_DIR / "best_model_card.json"
    with open(card_path) as f:
        card = json.load(f)

    preproc_config_path = BASE_DIR / "data" / "processed" / "best_preproc_config.json"
    with open(preproc_config_path) as f:
        preproc_config = json.load(f)

    clf, vectorizer, _ = load_model()

    cleaned = [preprocess_text(t, **preproc_config) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = clf.predict(X)

    results = []
    for text, pred in zip(texts, preds):
        label_str = "POSITIVE" if int(pred) == 1 else "NEGATIVE"
        results.append({
            "text": text,
            "label": label_str,
            "label_id": int(pred),
        })
    return results
