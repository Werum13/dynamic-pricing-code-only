import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "Elasticity")

import pandas as pd
import numpy as np
from DataPreprocessor import preprocessor
from ETL import etl_with_demand_target
from DemandModel import demand_model

# Read sample
df = pd.read_csv("/tmp/sample.csv",
                 usecols=["ITEMCODE","DATE_","UNITPRICE","TOTALPRICE","AMOUNT","CATEGORY1","CATEGORY2"])
cost_map = dict(pd.read_csv("data/cost.csv")[["ITEMCODE","cost"]].values)
df["cost"] = df["ITEMCODE"].map(cost_map)
print(f"Sample: {df.shape}")
print(df.head(3))

# 1. Preprocessor
df_pre = preprocessor(df.copy())
print(f"After preprocessor: {df_pre.shape}")

# 2. ETL
df_etl = etl_with_demand_target(df_pre)
print(f"After ETL: {df_etl.shape}")
print(f"Columns: {list(df_etl.columns)}")

# 3. Train model
TARGET_AND_FUTURE_COLS = [
    "DATE_","CATEGORY1","CATEGORY2",
    "GMV_1D","GMV_7D","GMV_15D","GMV_30D",
    "AMOUNT_7D_target","AMOUNT_7D","AMOUNT_15D","AMOUNT_30D",
]
max_date = df_etl["DATE_"].max()
cutoff   = max_date - pd.Timedelta(days=7)
train_full    = df_etl[df_etl["DATE_"] < cutoff].copy()
target_series = train_full["AMOUNT_7D_target"].copy()
train_df      = train_full.drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore")
valid         = train_df.notna().all(axis=1) & target_series.notna()
train_df, target_series = train_df[valid], target_series[valid]
print(f"Training rows: {len(train_df)}")

model = demand_model(train=train_df, target=target_series)
print("Model trained OK")
print(f"Model columns ({len(model.columns_)}): {model.columns_}")

# 4. Compute elasticity for first 5 dates
rows_out = []
for _, row in df_etl.iterrows():
    date = row["DATE_"].normalize().date()
    tmpl = pd.DataFrame([row]).drop(columns=TARGET_AND_FUTURE_COLS, errors="ignore").copy()
    tmpl["Id"] = 0

    feat_cols = [c for c in model.columns_ if c in tmpl.columns]
    if len(feat_cols) < len(model.columns_):
        continue
    if tmpl[feat_cols].isna().any(axis=1).iloc[0]:
        continue

    eps = model.elasticity(tmpl)
    rows_out.append({"DATE_": date, "ITEMCODE": int(row["ITEMCODE"]), "elasticity": round(float(eps[0]), 4)})

print(pd.DataFrame(rows_out).to_string(index=False))
