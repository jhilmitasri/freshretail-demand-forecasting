# ingest_flatten.py
"""
Stream FreshRetailNet-50K from HuggingFace, flatten hourly arrays, and save parquet chunks.
"""
import argparse
import os
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

def flatten_record(rec: dict) -> pd.DataFrame:
    """Turn one product-day record into 24 hourly rows."""
    base = {
        'city_id': rec['city_id'],
        'store_id': rec['store_id'],
        'management_group_id': rec['management_group_id'],
        'first_category_id': rec['first_category_id'],
        'second_category_id': rec['second_category_id'],
        'third_category_id': rec['third_category_id'],
        'product_id': rec['product_id'],
        'dt': rec['dt'],
        'discount': rec['discount'],
        'holiday_flag': rec['holiday_flag'],
        'activity_flag': rec['activity_flag'],
        'precpt': rec['precpt'],
        'avg_temperature': rec['avg_temperature'],
        'avg_humidity': rec['avg_humidity'],
        'avg_wind_level': rec['avg_wind_level'],
    }
    rows = []
    for hour, (sale, oos) in enumerate(zip(rec['hours_sale'], rec['hours_stock_status'])):
        row = base.copy()
        row.update({
            'hour': hour,
            'hourly_sale': sale,
            'hourly_stockout': oos,
        })
        rows.append(row)
    return pd.DataFrame(rows)

def main(args):
    ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K", split="train", streaming=True)
    os.makedirs(args.output_dir, exist_ok=True)
    buffer, chunk_idx = [], 0

    for rec in tqdm(ds, desc="Streaming records"):
        buffer.append(flatten_record(rec))
        if len(buffer) >= args.batch_size:
            df_chunk = pd.concat(buffer, ignore_index=True)
            path = os.path.join(args.output_dir, f"chunk_{chunk_idx:04d}.parquet")
            df_chunk.to_parquet(path, index=False)
            print(f"✅ Saved {path} ({df_chunk.shape[0]} rows)")
            chunk_idx += 1
            buffer = []

    if buffer:
        df_chunk = pd.concat(buffer, ignore_index=True)
        path = os.path.join(args.output_dir, f"chunk_{chunk_idx:04d}.parquet")
        df_chunk.to_parquet(path, index=False)
        print(f"✅ Saved {path} ({df_chunk.shape[0]} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of product-day records to flatten per chunk")
    parser.add_argument("--output-dir", type=str, default="data/flattened_chunks",
                        help="Where to write flattened parquet files")
    args = parser.parse_args()
    main(args)