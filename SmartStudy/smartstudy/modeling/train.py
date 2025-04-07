from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import pandas as pd
from tabpfn import TabPFNRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

from smartstudy.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "train_features.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "train_labels.csv",
    model_path: Path = MODELS_DIR / "tabpfn.pkl",
    # -----------------------------------------
):
    # ---- TabPFN ----
    logger.info("Training some model...")
    logger.info(f"Features path: {features_path}")
    logger.info(f"Labels path: {labels_path}")
    logger.info(f"Model path: {model_path}")
    logger.info("Loading data...")

    # Loading data
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    logger.info("Data loaded successfully.")

    # Training model
    logger.info("Training model...")
    model = TabPFNRegressor(random_state=42)
    model.fit(features, labels)
    logger.info("Model trained successfully.")
    
    # Saving model
    logger.info("Saving model...")
    joblib.dump(model, model_path)
    logger.info("Model saved successfully.")

    logger.success("Modeling training complete.")

    # -----------------------------------------
    # ------ Bayesian Opt ---------------------
    
    

    # -----------------------------------------
    # ------ Weighted KNN ---------------------

    # -----------------------------------------


if __name__ == "__main__":
    app()
