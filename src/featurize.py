# featurize.py
"""
Generate time-series features (lags, rolling, calendar) on daily-imputed dataset.
"""
import argparse
import os
import pandas as pd

def main(args):
    df = pd.read_parquet(args.input_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(
        ["third_category_id","dt"]
    ).reset_index(drop=True)

    # Calendar features
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    df["time_idx"] = (df["dt"] - df["dt"].min()).dt.days

    # Lags & rolling
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = (
            df.groupby("third_category_id")["daily_sale_imputed"]
              .shift(lag)
        )
    df["roll_mean_7"] = (
        df.groupby("third_category_id")["daily_sale_imputed"]
          .shift(1)
          .rolling(7)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # Drop rows with nulls in new features
    to_drop = [f"lag_{l}" for l in [1,7,14]] + ["roll_mean_7"]
    df = df.dropna(subset=to_drop)

    # Select columns for modeling
    feat_cols = [
        "third_category_id","dt","daily_sale_imputed",
        "lag_1","lag_7","lag_14","roll_mean_7",
        "day_of_week","is_weekend","time_idx","oos_hours_total"
    ]
    df_model = df[feat_cols]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_model.to_parquet(args.output_path, index=False)
    print(f"✅ Model-ready data saved: {df_model.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",  type=str,
                        default="data/daily_dataset/daily_df_imputed.parquet")
    parser.add_argument("--output-path", type=str,
                        default="data/daily_dataset/daily_df_modelready.parquet")
    args = parser.parse_args()
    main(args)