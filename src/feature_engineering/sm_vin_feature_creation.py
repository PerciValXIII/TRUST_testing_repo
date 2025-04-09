import pandas as pd
import numpy as np
from pathlib import Path
import os


class VintageFeatureEngineer:
    def __init__(self, application_path: str, previous_app_path: str, bureau_path: str):
        self.application_path = application_path
        self.previous_app_path = previous_app_path
        self.bureau_path = bureau_path

        self.df_application = None
        self.df_previous_application = None
        self.df_bureau = None
        self.features = None

        self.aggregators = ['sum', 'mean', 'max', 'min', 'std']
        self.time_periods = [3, 6, 12, 24, 36]

    def load_data(self):
        # Load raw CSVs into pandas DataFrames
        self.df_application = pd.read_csv(self.application_path)
        self.df_previous_application = pd.read_csv(self.previous_app_path)
        self.df_bureau = pd.read_csv(self.bureau_path)

    def generic_aggregator(self, df, groupby_col, agg_col, agg_funcs):
        # Generic aggregation function by group and column
        aggregated = df.groupby(groupby_col).agg({agg_col: agg_funcs}).reset_index()
        aggregated.columns = ['_'.join(col).rstrip('_') for col in aggregated.columns.values]
        return aggregated

    def vin_months_since_last_approved(self):
        # Calculates months since last approved loan using DAYS_DECISION
        df = self.df_previous_application[self.df_previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
        agg = self.generic_aggregator(df, 'SK_ID_CURR', 'DAYS_DECISION', ['max'])
        agg['vin_months_since_last_approved'] = agg['DAYS_DECISION_max'] / -30
        return agg[['SK_ID_CURR', 'vin_months_since_last_approved']]

    def vin_days_since_last_rejection(self):
        # Calculates days since last loan rejection based on DAYS_DECISION
        df = self.df_previous_application[self.df_previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
        agg = self.generic_aggregator(df, 'SK_ID_CURR', 'DAYS_DECISION', self.aggregators)
        for func in self.aggregators:
            agg[f'days_since_last_rejection_{func}'] = agg[f'DAYS_DECISION_{func}'] * -1
        return agg[['SK_ID_CURR'] + [col for col in agg.columns if col.startswith('days_since_last_rejection_')]]
    
    

    def vin_days_since_last_default(self):
        # Calculates days since last payment default using CREDIT_DAY_OVERDUE and DAYS_CREDIT
        df = self.df_bureau[self.df_bureau['CREDIT_DAY_OVERDUE'] > 0]
        agg = self.generic_aggregator(df, 'SK_ID_CURR', 'DAYS_CREDIT', ['min'])
        agg['days_since_last_default_min'] = agg['DAYS_CREDIT_min'] * -1
        return agg[['SK_ID_CURR', 'days_since_last_default_min']]

    def vin_days_since_last_loan_closure(self):
        # Calculates days since last loan closure using DAYS_ENDDATE_FACT
        df = self.df_bureau[self.df_bureau['CREDIT_ACTIVE'] == 'Closed']
        agg = self.generic_aggregator(df, 'SK_ID_CURR', 'DAYS_ENDDATE_FACT', self.aggregators)
        for func in self.aggregators:
            agg[f'vin_days_since_last_loan_closure_{func}'] = agg[f'DAYS_ENDDATE_FACT_{func}'] * -1
        return agg[['SK_ID_CURR'] + [col for col in agg.columns if col.startswith('vin_days_since_last_loan_closure_')]]

    def vin_days_since_first_loan_taken(self):
        # Calculates days since first loan was taken using DAYS_CREDIT
        agg = self.generic_aggregator(self.df_bureau, 'SK_ID_CURR', 'DAYS_CREDIT', self.aggregators)
        for func in self.aggregators:
            agg[f'vin_days_since_first_loan_taken_{func}'] = agg[f'DAYS_CREDIT_{func}'] * -1
        return agg[['SK_ID_CURR'] + [col for col in agg.columns if col.startswith('vin_days_since_first_loan_taken_')]]

    # def vin_avg_installment_income_ratio(self):
    #     # Calculates the ratio of AMT_ANNUITY to AMT_INCOME_TOTAL
    #     df = self.df_application.copy()
    #     df['vin_avg_installment_income_ratio'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    #     return df[['SK_ID_CURR', 'vin_avg_installment_income_ratio']]

    def engineer_features(self):
        # Main method to engineer all vintage features and merge them
        df_features = self.df_application[['SK_ID_CURR']].copy()

        funcs = [
            self.vin_months_since_last_approved,
            self.vin_days_since_last_rejection,
            self.vin_days_since_last_default,
            self.vin_days_since_last_loan_closure,
            self.vin_days_since_first_loan_taken
            # self.vin_avg_installment_income_ratio
        ]

        for func in funcs:
            df = func()
            df_features = df_features.merge(df, on='SK_ID_CURR', how='left')

        # Join TARGET
        df_target = self.df_application[['SK_ID_CURR', 'TARGET']]
        self.features = df_features.merge(df_target, on='SK_ID_CURR', how='left')

    def get_features(self):
        # Returns final engineered DataFrame
        if self.features is None:
            raise ValueError("Features not engineered yet. Run engineer_features().")
        return self.features


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[2]

    engineer = VintageFeatureEngineer(
        application_path=os.path.join(root_dir, "data", "raw", "application_train.csv"),
        previous_app_path=os.path.join(root_dir, "data", "raw", "previous_application.csv"),
        bureau_path=os.path.join(root_dir, "data", "raw", "bureau.csv")
    )
    engineer.load_data()
    engineer.engineer_features()
    features_df = engineer.get_features()

    output_path = os.path.join(root_dir, "data", "processed", "vin_features.csv")
    features_df.to_csv(output_path, index=False)