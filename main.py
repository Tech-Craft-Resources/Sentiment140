"""
Entry point shortcuts for the Sentiment140 project.

Usage:
    python main.py serve     — Start the FastAPI inference API
    python main.py mlflow    — Start a local MLflow tracking server
"""
import sys
import subprocess


def serve():
    subprocess.run(
        ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd="api",
    )


def start_mlflow():
    subprocess.run(
        ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
    )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "help"
    if command == "serve":
        serve()
    elif command == "mlflow":
        start_mlflow()
    else:
        print(__doc__)
