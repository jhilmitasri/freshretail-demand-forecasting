# train_pipeline.py

#!/usr/bin/env python
import os
import argparse
from ingest_flatten import stream_and_flatten
from aggregate_impute import aggregate_and_impute
from featurize import build_features
from train_darts_nbeats import main as train_nbeats

def main():
    p = argparse.ArgumentParser(
        description="Full pipeline: ingest → aggregate/impute → featurize → train N-BEATS"
    )
    p.add_argument("--split", choices=["train","eval"], default="train")
    p.add_argument("--batch-size", type=int, default=12000)
    p.add_argument(
        "--flat-dir",
        default="data/flattened_chunks",
        help="where to write/read hourly parquet chunks"
    )
    p.add_argument(
        "--daily-path",
        default="data/daily_dataset/daily_df_imputed.parquet",
        help="where to write/read daily aggregated+imputed parquet"
    )
    p.add_argument(
        "--modelready-path",
        default="data/daily_dataset/daily_df_modelready.parquet",
        help="where to write/read final feature‐ready parquet"
    )
    p.add_argument(
        "--cats",
        nargs="+",
        type=int,
        required=True,
        help="third_category_id list to train"
    )
    p.add_argument(
        "--model-dir",
        default="models",
        help="where to save trained N-BEATS models"
    )
    p.add_argument(
        "--input-len",
        type=int,
        default=28,
        help="NBEATS input_chunk_length"
    )
    p.add_argument(
        "--output-len",
        type=int,
        default=7,
        help="NBEATS output_chunk_length"
    )
    args = p.parse_args()

    # 1️⃣ Ingest & flatten hourly → parquet chunks
    print("\n▶️  Step 1: stream & flatten")
    os.makedirs(args.flat_dir, exist_ok=True)
    stream_and_flatten(
        split=args.split,
        batch_size=args.batch_size,
        output_dir=args.flat_dir
    )

    # 2️⃣ Aggregate & impute → daily parquet
    print("\n▶️  Step 2: aggregate & impute")
    aggregate_and_impute(
        input_dir=args.flat_dir,
        output_path=args.daily_path
    )

    # 3️⃣ Build features → model‐ready parquet
    print("\n▶️  Step 3: featurize")
    build_features(
        input_path=args.daily_path,
        output_path=args.modelready_path
    )

    # 4️⃣ Train per‐category N-BEATS
    print("\n▶️  Step 4: train N-BEATS models")
    train_nbeats(
    cats=args.cats,
    modelready_path=args.modelready_path,
    model_dir=args.model_dir,
    input_len=args.input_len,
    output_len=args.output_len
)

if __name__ == "__main__":
    main()