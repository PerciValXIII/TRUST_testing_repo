# Import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import gc
import logging
import os

# Enable garbage collection
gc.enable()

# # Set up logging for cloud execution tracking
# logging.basicConfig(filename="stacking_log.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define parameters
NFOLDS = 5  
SEED = 42
N_JOBS = -1  # Use all available cores
MODEL_DIR = "/models" 

# Load dataset 
filepath = "../data/appl_train_benchmark_001.csv"
data = pd.read_csv(filepath)  

# Separate features and target
y = data['TARGET']
X = data.drop(columns=['TARGET'])

X_train, X_final_test, y_train, y_final_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)

# Log dataset details
logging.info(f"Original dataset shape: {data.shape}")
logging.info(f"Training dataset shape: {X_train.shape}, Final test set shape: {X_final_test.shape}")

# Define cross-validation
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)

# Define models and hyperparameters
model_configs = {
    "XGBoost": {
        "model": xgb.XGBClassifier,
        "params": {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "n_jobs": N_JOBS,
            "random_state": SEED
        }
    },
    "LightGBM": {
        "model": LGBMClassifier,
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "num_leaves": 123,
            "colsample_bytree": 0.8,
            "subsample": 0.9,
            "max_depth": 15,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": N_JOBS,
            "random_state": SEED
        }
    },
    "CatBoost": {
        "model": CatBoostClassifier,
        "params": {
            "iterations": 200,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 40,
            "bootstrap_type": "Bernoulli",
            "subsample": 0.7,
            "scale_pos_weight": 5,
            "eval_metric": "AUC",
            "verbose": 0,
            "random_seed": SEED
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 12,
            "max_features": 0.2,
            "min_samples_leaf": 2,
            "n_jobs": N_JOBS,
            "random_state": SEED
        }
    },
    "ExtraTrees": {
        "model": ExtraTreesClassifier,
        "params": {
            "n_estimators": 200,
            "max_depth": 12,
            "max_features": 0.5,
            "min_samples_leaf": 2,
            "n_jobs": N_JOBS,
            "random_state": SEED
        }
    }
}



# Function to generate out-of-fold (OOF) predictions
def get_oof_predictions(model_name, model_class, params, X_train, y_train, X_test):
    """
    Generates Out-of-Fold (OOF) predictions using KFold CV.
    """
    oof_train = np.zeros(X_train.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    oof_test_skf = np.empty((NFOLDS, X_test.shape[0]))

    for i, (train_idx, valid_idx) in enumerate(kf.split(X_train)):
        logging.info(f"Training fold {i+1} for {model_name}...")

        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model = model_class(**params)
        model.fit(X_tr, y_tr)

        oof_train[valid_idx] = model.predict_proba(X_val)[:,1]
        oof_test_skf[i, :] = model.predict_proba(X_test)[:,1]

        # Save each trained model
        joblib.dump(model, os.path.join(MODEL_DIR, f"{model_name}_fold_{i+1}.pkl"))

    oof_test[:] = oof_test_skf.mean(axis=0)

    logging.info(f"{model_name} CV AUC Score: {roc_auc_score(y_train, oof_train):.5f}")
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


oof_train_list, oof_test_list = [], []

# Run models in parallel using joblib
for model_name, config in model_configs.items():
    oof_train, oof_test = get_oof_predictions(model_name, config["model"], config["params"], X_train, y_train, X_test)
    oof_train_list.append(oof_train)
    oof_test_list.append(oof_test)

# Stack OOF predictions
stacked_train = np.hstack(oof_train_list)
stacked_test = np.hstack(oof_test_list)

logging.info(f"Stacked train shape: {stacked_train.shape}, Stacked test shape: {stacked_test.shape}")

# Train meta-model (Logistic Regression)
logging.info("Training meta-model...")
meta_model = LogisticRegression()
meta_model.fit(stacked_train, y_train)

# Make final predictions
final_predictions = meta_model.predict_proba(stacked_test)[:,1]

# Save meta-model and predictions
joblib.dump(meta_model, os.path.join(MODEL_DIR, "stacking_meta_model.pkl"))
pd.DataFrame({"ID": np.arange(len(final_predictions)), "TARGET": final_predictions}).to_csv("stacking_predictions.csv", index=False)

logging.info("Stacking model training completed. Predictions saved.")

# Save final hold-out set for evaluation later
X_final_test.to_csv("final_test_features.csv", index=False)
y_final_test.to_csv("final_test_labels.csv", index=False)
logging.info("Final test set saved for future evaluation.")

