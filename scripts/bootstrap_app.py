from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.getcwd())

from src.download_data import main as download_data
from src.train import train


CONFIG_PATH = "configs/config.yaml"
DATA_PATH = "data/aapl_daily.csv"
MODEL_PATH = "models/best_model.joblib"


def ensure_artifacts() -> None:
    if not os.path.exists(DATA_PATH):
        print("AAPL data not found. Downloading market history...")
        download_data(CONFIG_PATH)
    else:
        print(f"Using existing data at {DATA_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("Model artifact not found. Training models before app startup...")
        train(CONFIG_PATH)
    else:
        print(f"Using existing model at {MODEL_PATH}")


def main() -> None:
    ensure_artifacts()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "src/app.py",
        "--server.address=0.0.0.0",
        "--server.port=8501",
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
