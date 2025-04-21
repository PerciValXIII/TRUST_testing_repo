import pandas as pd
import logging
import time
import os
from contextlib import contextmanager
from src.model_training.base_model_utils import train_base_models_with_oof
from src.model_training.meta_model import train_meta_model

# --- Logging Setup ---
os.makedirs("outputs/logs", exist_ok=True)
log_file = "outputs/logs/model_training_1.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Utility Context Manager for Time Tracking ---
@contextmanager
def log_stage(name):
    logger.info(f"🔄 Starting: {name}")
    start_time = time.time()
    yield
    logger.info(f"✅ Completed: {name} in {time.time() - start_time:.2f} seconds\n")

# --- Main Pipeline ---
if __name__ == "__main__":
    logger.info("🚀 Starting model stacking pipeline")

    with log_stage("Loading Processed Datasets"):
        demog_df = pd.read_csv("data/processed/demog_features_baseline_ready.csv")
        deq_df = pd.read_csv("data/processed/deq_features_baseline_ready.csv")
        vin_df = pd.read_csv("data/processed/vintage_features_baseline_ready.csv")

    with log_stage("Demographic Base Models Training"):
        demog_preds, demog_target = train_base_models_with_oof(demog_df, "demog", logger=logger)

    with log_stage("Delinquency Base Models Training"):
        deq_preds, _ = train_base_models_with_oof(deq_df, "deq", logger=logger)

    with log_stage("Vintage Base Models Training"):
        vin_preds, _ = train_base_models_with_oof(vin_df, "vin", logger=logger)

    with log_stage("Target Table Creation"):
        target_df = pd.concat([
            demog_df[["SK_ID_CURR", "TARGET"]],
            deq_df[["SK_ID_CURR", "TARGET"]],
            vin_df[["SK_ID_CURR", "TARGET"]]
        ]).drop_duplicates("SK_ID_CURR")

    with log_stage("Meta Model Training"):
        train_meta_model(demog_preds, deq_preds, vin_preds, target_df, logger=logger)

    logger.info("🎯 Model stacking pipeline completed successfully.")