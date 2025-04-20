from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def train_meta_model(demog_preds, deq_preds, vin_preds, target_df):
    meta_df = demog_preds.merge(deq_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(vin_preds, on="SK_ID_CURR", how="outer")
    meta_df = meta_df.merge(target_df, on="SK_ID_CURR", how="left")
    meta_df = meta_df.dropna()

    X_meta = meta_df.drop(columns=["SK_ID_CURR", "TARGET"])
    y_meta = meta_df["TARGET"]

    model = LogisticRegression(max_iter=1000)
    scores = cross_val_score(model, X_meta, y_meta, cv=5, scoring="roc_auc")
    print(f"[META] Logistic Regression AUC (CV): {scores.mean():.4f}")
    return model.fit(X_meta, y_meta)