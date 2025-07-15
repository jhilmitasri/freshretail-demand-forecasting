# aggregate_impute.py
"""
Chunk-wise aggregation of hourly sales into daily sales with imputation.
"""
import argparse
import os
import glob
import pandas as pd

def main(args):
    ID_COLS = [
        "city_id","store_id","management_group_id",
        "first_category_id","second_category_id","third_category_id",
        "product_id","dt"
    ]
    daily_chunks = []
    for fp in sorted(glob.glob(f"{args.input_dir}/*.parquet")):
        df = pd.read_parquet(fp)
        # Raw daily sums & mean contextual
        raw = df.groupby(ID_COLS, as_index=False).agg(
            raw_sale       = ("hourly_sale",     "sum"),
            oos_hours_total= ("hourly_stockout", "sum"),
            discount       = ("discount",        "mean"),
            holiday_flag   = ("holiday_flag",    "max"),
            activity_flag  = ("activity_flag",   "max"),
            precpt         = ("precpt",          "mean"),
            avg_temperature= ("avg_temperature", "mean"),
            avg_humidity   = ("avg_humidity",    "mean"),
            avg_wind_level = ("avg_wind_level",  "mean"),
        )
        # In-stock sums & counts
        instock = (
            df[df["hourly_stockout"] == 0]
              .groupby(ID_COLS, as_index=False)
              .agg(instock_sum=("hourly_sale","sum"),
                   instock_count=("hourly_sale","count"))
        )
        # Merge and compute imputed daily sale
        agg = raw.merge(instock, on=ID_COLS, how="left")
        agg["instock_count"] = agg["instock_count"].fillna(0)
        agg["instock_mean"] = agg.apply(
            lambda r: r.instock_sum / r.instock_count if r.instock_count>0 else 0,
            axis=1
        )
        agg["daily_sale_imputed"] = (
            agg["raw_sale"] + agg["instock_mean"] * agg["oos_hours_total"]
        )
        daily_chunks.append(agg)

        if len(daily_chunks) % 50 == 0:
            print(f"🔄 Processed {len(daily_chunks)} chunks…")

    # Combine and save
    daily_df = pd.concat(daily_chunks, ignore_index=True)
    keep_cols = ID_COLS + [
        "daily_sale_imputed","oos_hours_total",
        "discount","holiday_flag","activity_flag",
        "precpt","avg_temperature","avg_humidity","avg_wind_level"
    ]
    daily_df = daily_df[keep_cols]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    daily_df.to_parquet(args.output_path, index=False)
    print(f"✅ Completed: {daily_df.shape[0]:,} daily rows → {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",   type=str, default="data/flattened_chunks",
                        help="Hourly parquet chunks dir")
    parser.add_argument("--output-path", type=str,
                        default="data/daily_dataset/daily_df_imputed.parquet",
                        help="Where to write daily-imputed parquet")
    args = parser.parse_args()
    main(args)