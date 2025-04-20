import pandas as pd
from src.feature_selection.preprocess_pipeline import Preprocessor

# Load your data
df_raw = pd.read_csv('data/raw/application_train.csv')

# Run preprocessing
preprocessor = Preprocessor(df_raw)
df_processed = preprocessor.run()

# Now you can use df_processed for modeling
print(df_processed.head())
print(df_processed.info())
print(df_processed.isna().sum())

#save in csv
df_processed.to_csv("data/processed/demog_features_baseline_ready.csv", index=False)