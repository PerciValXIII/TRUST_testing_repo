import pandas as pd
import numpy as np
import os
from pathlib import Path

class AMTFFeatureEngineer:
    def __init__(self, app_path, pos_path, cc_path, inst_path):
        self.app_path = app_path
        self.pos_path = pos_path
        self.cc_path = cc_path
        self.inst_path = inst_path

        self.df_app = None
        self.df_pos = None
        self.df_cc = None
        self.df_inst = None
        self.features = None

    def load_data(self):
        self.df_app = pd.read_csv(self.app_path)
        self.df_pos = pd.read_csv(self.pos_path)
        self.df_cc = pd.read_csv(self.cc_path)
        self.df_inst = pd.read_csv(self.inst_path)

    def reduce_and_label_sources(self):
        app_ids = self.df_app["SK_ID_CURR"].unique()

        self.df_pos = self.df_pos[self.df_pos["SK_ID_CURR"].isin(app_ids)].copy()
        self.df_pos["NAME_CONTRACT_TYPE"] = "Cash loans"

        self.df_cc = self.df_cc[self.df_cc["SK_ID_CURR"].isin(app_ids)].copy()
        self.df_cc["NAME_CONTRACT_TYPE"] = "Revolving loans"

        df_inst = self.df_inst[self.df_inst["SK_ID_CURR"].isin(app_ids)].copy()
        df_inst = df_inst.merge(
            self.df_pos[["SK_ID_CURR", "SK_ID_PREV"]],
            how="left", on=["SK_ID_CURR", "SK_ID_PREV"], indicator="pos_ind"
        )
        df_inst = df_inst.merge(
            self.df_cc[["SK_ID_CURR", "SK_ID_PREV"]],
            how="left", on=["SK_ID_CURR", "SK_ID_PREV"], indicator="cc_ind"
        )

        df_inst["NAME_CONTRACT_TYPE"] = np.select(
            [
                df_inst["pos_ind"] == "both",
                df_inst["cc_ind"] == "both"
            ],
            [
                "Cash loans",
                "Revolving loans"
            ],
            default=None
        )

        self.df_inst = df_inst.drop(columns=["pos_ind", "cc_ind"])

    def group_stats(self, df, group_cols, aggregations):
        grouped = df.groupby(group_cols).agg(aggregations)
        grouped.columns = ['amtf_' + '_'.join(col).strip() for col in grouped.columns.values]
        return grouped.reset_index()

    def engineer_pos_features(self):
        df = self.df_pos.copy()
        df["amtf_loan_duration_months"] = df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].transform("max") - df.groupby("SK_ID_PREV")["MONTHS_BALANCE"].transform("min") + 1
        df["amtf_instalments_completed"] = df["CNT_INSTALMENT"] - df["CNT_INSTALMENT_FUTURE"]
        df["amtf_future_installment_ratio"] = df["CNT_INSTALMENT_FUTURE"] / df["CNT_INSTALMENT"]
        df["amtf_overdue_flag"] = (df["SK_DPD"] > 0).astype(int)
        df["amtf_deferral_flag"] = (df["SK_DPD_DEF"] > 0).astype(int)
        df["amtf_active_status_flag"] = (df["NAME_CONTRACT_STATUS"] == "Active").astype(int)
        df["amtf_completed_status_flag"] = (df["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

        agg = self.group_stats(
            df,
            ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE"],
            {
                "MONTHS_BALANCE": ["max", "min", "mean", "std"],
                "CNT_INSTALMENT": ["max", "min", "mean", "std", "sum"],
                "CNT_INSTALMENT_FUTURE": ["max", "min", "mean", "std", "sum"],
                "amtf_instalments_completed": ["sum"],
                "amtf_future_installment_ratio": ["mean"],
                "SK_DPD": ["max", "mean", "std"],
                "amtf_overdue_flag": ["sum"],
                "SK_DPD_DEF": ["max", "mean", "std"],
                "amtf_deferral_flag": ["sum"],
                "amtf_active_status_flag": ["sum", "mean"],
                "amtf_completed_status_flag": ["sum", "mean"],
            }
        )
        return agg

    def engineer_credit_card_features(self):
        df = self.df_cc.copy()
        df["amtf_credit_utilization_ratio"] = df["AMT_BALANCE"] / df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)
        df["amtf_overdue_flag"] = (df["SK_DPD"] > 0).astype(int)
        df["amtf_deferral_flag"] = (df["SK_DPD_DEF"] > 0).astype(int)
        df["amtf_active_status_flag"] = (df["NAME_CONTRACT_STATUS"] == "Active").astype(int)
        df["amtf_completed_status_flag"] = (df["NAME_CONTRACT_STATUS"] == "Completed").astype(int)

        agg = self.group_stats(
            df,
            ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE"],
            {
                "AMT_BALANCE": ["max", "min", "mean", "std", "sum"],
                "AMT_CREDIT_LIMIT_ACTUAL": ["max", "min", "mean", "sum"],
                "amtf_credit_utilization_ratio": ["mean", "max"],
                "AMT_PAYMENT_CURRENT": ["max", "min", "mean", "sum"],
                "AMT_PAYMENT_TOTAL_CURRENT": ["max", "sum"],
                "AMT_RECEIVABLE_PRINCIPAL": ["max", "min", "mean", "sum"],
                "AMT_RECIVABLE": ["mean", "sum"],
                "AMT_DRAWINGS_ATM_CURRENT": ["sum"],
                "AMT_DRAWINGS_POS_CURRENT": ["sum"],
                "AMT_DRAWINGS_OTHER_CURRENT": ["sum"],
                "AMT_DRAWINGS_CURRENT": ["sum"],
                "CNT_DRAWINGS_ATM_CURRENT": ["mean"],
                "CNT_DRAWINGS_POS_CURRENT": ["mean"],
                "CNT_DRAWINGS_CURRENT": ["mean"],
                "SK_DPD": ["max", "mean"],
                "amtf_overdue_flag": ["sum", "mean"],
                "SK_DPD_DEF": ["max", "mean"],
                "amtf_deferral_flag": ["sum", "mean"],
                "amtf_active_status_flag": ["sum", "mean"],
                "amtf_completed_status_flag": ["sum", "mean"]
            }
        )
        return agg

    def engineer_installment_features(self):
        df = self.df_inst.copy()
        df["amtf_payment_to_instalment_ratio"] = df["AMT_PAYMENT"] / df["AMT_INSTALMENT"].replace(0, np.nan)
        df["amtf_late_flag"] = (df["DAYS_ENTRY_PAYMENT"] > 0).astype(int)
        df["amtf_early_flag"] = (df["DAYS_ENTRY_PAYMENT"] < 0).astype(int)
        df["amtf_exact_or_over_flag"] = (df["AMT_PAYMENT"] >= df["AMT_INSTALMENT"]).astype(int)
        df["amtf_under_flag"] = (df["AMT_PAYMENT"] < df["AMT_INSTALMENT"]).astype(int)
        df["amtf_under_amt"] = (df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]).clip(lower=0)

        agg = self.group_stats(
            df,
            ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE"],
            {
                "AMT_INSTALMENT": ["sum", "mean", "max", "min"],
                "AMT_PAYMENT": ["sum", "mean", "max", "min", "std"],
                "amtf_payment_to_instalment_ratio": ["mean"],
                "amtf_late_flag": ["sum", "mean"],
                "amtf_early_flag": ["sum", "mean"],
                "DAYS_ENTRY_PAYMENT": ["std"],
                "DAYS_INSTALMENT": ["max", "min", "mean"],
                "NUM_INSTALMENT_VERSION": ["nunique"],
                "NUM_INSTALMENT_NUMBER": ["nunique"],
                "amtf_exact_or_over_flag": ["sum"],
                "amtf_under_flag": ["sum"],
                "amtf_under_amt": ["sum"]
            }
        )
        return agg

    def engineer_features(self):
        self.reduce_and_label_sources()

        pos_df = self.engineer_pos_features()
        cc_df = self.engineer_credit_card_features()
        inst_df = self.engineer_installment_features()

        print(f"POS features shape: {pos_df.shape}")
        print(f"Credit Card features shape: {cc_df.shape}")
        print(f"Installments features shape: {inst_df.shape}")

        combined = pd.merge(pos_df, cc_df, on=["SK_ID_CURR", "SK_ID_PREV"], how="outer")
        combined = pd.merge(combined, inst_df, on=["SK_ID_CURR", "SK_ID_PREV"], how="outer")

        self.features = combined

    def get_features(self) -> pd.DataFrame:
        if self.features is None:
            raise ValueError("Features have not been engineered yet.")
        return self.features


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]

    engineer = AMTFFeatureEngineer(
        app_path=os.path.join(root, "data", "raw", "application_train.csv"),
        pos_path=os.path.join(root, "data", "raw", "POS_CASH_balance.csv"),
        cc_path=os.path.join(root, "data", "raw", "credit_card_balance.csv"),
        inst_path=os.path.join(root, "data", "raw", "installments_payments.csv")
    )

    engineer.load_data()
    engineer.engineer_features()
    features_df = engineer.get_features()

    output_path = os.path.join(root, "data", "processed", "amtf_features_level1.csv")
    features_df.to_csv(output_path, index=False)
