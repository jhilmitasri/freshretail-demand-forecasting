#!/usr/bin/env python
import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def stream_and_flatten(split: str, batch_size: int, output_dir: str) -> None:
    """
    Streams the FreshRetailNet-50K dataset, flattens each record into one row per hour,
    and writes out Parquet chunks of size `batch_size`.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split=split, streaming=True)
    buffer = []
    count = 0
    # initialize progress bar for records and chunk count
    pbar = tqdm(ds, desc=f"Streaming & flattening ({split})", unit="records")
    chunk_idx = 0

    for rec in pbar:
        # pull out scalar fields
        meta = {
            "city_id": rec["city_id"],
            "store_id": rec["store_id"],
            "management_group_id": rec["management_group_id"],
            "first_category_id": rec["first_category_id"],
            "second_category_id": rec["second_category_id"],
            "third_category_id": rec["third_category_id"],
            "product_id": rec["product_id"],
            "dt": rec["dt"],
            "discount": rec["discount"],
            "activity_flag": rec["activity_flag"],
            "holiday_flag": rec["holiday_flag"],
            "precpt": rec["precpt"],
            "avg_temperature": rec["avg_temperature"],
            "avg_humidity": rec["avg_humidity"],
            "avg_wind_level": rec["avg_wind_level"],
        }
        for h, (sale, stockout) in enumerate(zip(rec["hours_sale"], rec["hours_stock_status"])):
            row = {
                **meta,
                "hour": h,
                "hourly_sale": sale,
                "hourly_stockout": stockout
            }
            buffer.append(row)
            count += 1

            if count >= batch_size:
                df = pd.DataFrame(buffer)
                path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.parquet")
                df.to_parquet(path, index=False)
                buffer.clear()
                count = 0
                chunk_idx += 1
                pbar.set_postfix(chunks=chunk_idx)

    # flush remainder
    if buffer:
        df = pd.DataFrame(buffer)
        path = os.path.join(output_dir, f"chunk_{chunk_idx:04d}.parquet")
        df.to_parquet(path, index=False)
        chunk_idx += 1
        pbar.set_postfix(chunks=chunk_idx)

def main():
    parser = argparse.ArgumentParser(
        description="Stream & flatten FreshRetailNet-50K into hourly Parquet chunks."
    )
    parser.add_argument(
        "--split", choices=["train", "eval"], default="train",
        help="Dataset split to stream."
    )
    parser.add_argument(
        "--batch-size", type=int, default=12000,
        help="Number of rows per Parquet chunk."
    )
    parser.add_argument(
        "--output-dir", default="data/flattened_chunks",
        help="Directory to write chunk_{:04d}.parquet files."
    )
    args = parser.parse_args()
    stream_and_flatten(args.split, args.batch_size, args.output_dir)

if __name__ == "__main__":
    main()