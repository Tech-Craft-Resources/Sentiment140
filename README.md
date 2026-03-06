# Sentiment140 — Análisis de Sentimientos

Laboratorio de análisis de sentimientos sobre el dataset [Sentiment140](https://huggingface.co/datasets/adilbekovich/Sentiment140Twitter).  
El objetivo es igualar o superar el desempeño de `pipeline('sentiment-analysis')` de Hugging Face (DistilBERT fine-tuned en SST-2) usando métodos clásicos de aprendizaje de máquina.

**Integrantes:** Nicolas Rodriguez · Daniel Velasco

---

## Estructura del proyecto

```
sentiment140/
├─ data/
│  ├─ raw/          ← Dataset original de HuggingFace (CSV)
│  ├─ processed/    ← Tweets preprocesados con spaCy/NLTK (parquet)
│  └─ encoded/      ← Matrices de características BoW / TF-IDF (npz)
├─ notebooks/
│  ├─ 01_preprocessing.ipynb   ← Limpieza y ablación de preprocesamiento
│  ├─ 02_encoding.ipynb        ← BoW, TF-IDF unigrama y bigrama
│  ├─ 03_baseline.ipynb        ← Baseline: Multinomial NB + BoW
│  ├─ 04_models_nicolas.ipynb  ← Random Forest + MLP (Nicolas)
│  ├─ 04_models_daniel.ipynb   ← Logistic Regression + SVM (Daniel)
│  └─ 05_distilbert.ipynb      ← Comparación con DistilBERT HF
├─ models/
│  ├─ best_model.pkl           ← Mejor pipeline sklearn serializado
│  └─ best_model_card.json     ← Descripción del modelo
├─ api/
│  ├─ main.py                  ← FastAPI app con 5 endpoints
│  ├─ schemas.py               ← Modelos Pydantic
│  ├─ inference.py             ← Carga del modelo y predicción
│  └─ requirements.txt
├─ requirements.txt
├─ .env.example
└─ pyproject.toml
```

---

## Setup

### 1. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Descargar modelos spaCy y NLTK

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 3. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env con tu MLFLOW_TRACKING_URI si usas servidor remoto
```

### 4. Iniciar MLflow (local)

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Abrir [http://localhost:5000](http://localhost:5000) para ver los experimentos.

### 5. Ejecutar los notebooks en orden

```
01_preprocessing → 02_encoding → 03_baseline → 04_models_* → 05_distilbert
```

### 6. Levantar la API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Documentación automática en [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Métrica principal

**F1-score macro** — también se registran accuracy, precision y recall en MLflow.

## Distribución de experimentos

| Integrante | Modelos |
|---|---|
| Nicolas Rodriguez | Random Forest, MLPClassifier |
| Daniel Velasco | Logistic Regression, LinearSVC |
