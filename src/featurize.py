#!/usr/bin/env python
import os
import argparse
import pandas as pd
from tqdm import tqdm

def build_features(input_path: str, output_path: str) -> None:
    """
    Reads daily_df_imputed.parquet, creates lags, rolling, calendar features,
    drops NaNs, and writes out a model-ready Parquet.
    """
    df = pd.read_parquet(input_path)
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["third_category_id", "dt"])

    # calendar
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    # relative time index
    df["time_idx"] = (df["dt"] - df["dt"].min()).dt.days

    # lags
    for lag in [1, 7, 14]:
        df[f"lag_{lag}"] = (
            df.groupby("third_category_id")["daily_sale_imputed"]
              .shift(lag)
        )

    # rolling mean
    df["roll_mean_7"] = (
        df.groupby("third_category_id")["daily_sale_imputed"]
          .shift(1)
          .rolling(7)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # drop initial NaNs
    df_model = df.dropna()

    # write out
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_model.to_parquet(output_path, index=False)

def main():
    parser = argparse.ArgumentParser(
        description="Build time-series features on daily imputed dataset."
    )
    parser.add_argument(
        "--input-path", default="src/data/daily_dataset/daily_df_imputed.parquet",
        help="Path to the aggregated+imputed daily Parquet."
    )
    parser.add_argument(
        "--output-path", default="src/data/daily_dataset/daily_df_modelready.parquet",
        help="Path for the model-ready Parquet."
    )
    args = parser.parse_args()
    build_features(args.input_path, args.output_path)

if __name__ == "__main__":
    main()