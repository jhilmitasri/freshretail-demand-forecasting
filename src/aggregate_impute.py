#!/usr/bin/env python
import os
import glob
import argparse
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# columns that define one product-store-day
ID_COLS = [
    "city_id", "store_id", "management_group_id",
    "first_category_id", "second_category_id", "third_category_id",
    "product_id", "dt"
]

def aggregate_and_impute(input_dir: str, output_path: str) -> None:
    """
    Reads each hourly Parquet chunk, aggregates to daily, imputes stockout hours,
    and writes/appends rows to a single daily Parquet file.
    """
    files = sorted(glob.glob(f"{input_dir}/*.parquet"))
    writer = None

    for fp in tqdm(files, desc="Aggregating daily chunks", unit="file"):
        df = pd.read_parquet(fp)

        # 1️⃣ Raw daily aggregates
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

        # 2️⃣ In-stock aggregates
        instock = (
            df[df["hourly_stockout"] == 0]
            .groupby(ID_COLS, as_index=False)
            .agg(
                instock_sum   = ("hourly_sale", "sum"),
                instock_count = ("hourly_sale", "count"),
            )
        )

        # 3️⃣ Merge, compute imputed daily sale
        agg = raw.merge(instock, on=ID_COLS, how="left")
        agg["instock_count"] = agg["instock_count"].fillna(0)
        # avoid zero‐division
        agg["instock_mean"] = agg.apply(
            lambda r: r.instock_sum / r.instock_count if r.instock_count > 0 else 0.0,
            axis=1
        )
        agg["daily_sale_imputed"] = (
            agg["raw_sale"] + agg["instock_mean"] * agg["oos_hours_total"]
        )

        # select final columns and avoid SettingWithCopyWarning
        daily = agg[ID_COLS + [
            "daily_sale_imputed", "oos_hours_total",
            "discount", "holiday_flag", "activity_flag",
            "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"
        ]].copy()
        # enforce 64-bit ints to match existing Parquet schema
        daily["holiday_flag"]  = daily["holiday_flag"].astype("int64")
        daily["activity_flag"] = daily["activity_flag"].astype("int64")

        # write/app​end to Parquet
        table = pa.Table.from_pandas(daily, preserve_index=False)
        if writer is None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate & impute hourly Parquet chunks into daily Parquet."
    )
    parser.add_argument(
        "--input-dir", default="data/flattened_chunks",
        help="Directory containing chunk_*.parquet."
    )
    parser.add_argument(
        "--output-path", default="data/daily_dataset/daily_df_imputed.parquet",
        help="Path for the output daily Parquet."
    )
    args = parser.parse_args()
    aggregate_and_impute(args.input_dir, args.output_path)

if __name__ == "__main__":
    main()