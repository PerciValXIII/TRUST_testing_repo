import pandas as pd
from src.model_training.base_model_utils import train_base_models_with_oof
from src.model_training.meta_model import train_meta_model

# Load processed feature sets
demog_df = pd.read_csv("data/processed/demog_features_baseline_ready.csv")
deq_df = pd.read_csv("data/processed/deq_features_baseline_ready.csv")
vin_df = pd.read_csv("data/processed/vintage_features_baseline_ready.csv")

# Train base models and generate OOF predictions
demog_preds, demog_target = train_base_models_with_oof(demog_df, "demog")
deq_preds, _ = train_base_models_with_oof(deq_df, "deq")
vin_preds, _ = train_base_models_with_oof(vin_df, "vin")

# Build a master target table using SK_ID_CURR
target_df = pd.concat([
    demog_df[["SK_ID_CURR", "TARGET"]],
    deq_df[["SK_ID_CURR", "TARGET"]],
    vin_df[["SK_ID_CURR", "TARGET"]]
]).drop_duplicates("SK_ID_CURR")

# Final meta-model training
train_meta_model(demog_preds, deq_preds, vin_preds, target_df)