#!/usr/bin/env python

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from ingest_flatten import stream_and_flatten
from aggregate_impute import aggregate_and_impute
from featurize import build_features
import torch
from darts import TimeSeries
from darts.models.forecasting.nbeats import NBEATSModel

def main():
    parser = argparse.ArgumentParser(
        description="Prediction pipeline: eval feature prep, N-BEATS forecast, actual vs. predicted plot"
    )
    # eval feature preparation
    parser.add_argument("--batch-size", type=int, default=12000,
                        help="Batch size for flattening")
    parser.add_argument("--flat-dir", default="data/flattened_chunks_eval",
                        help="Where to write/read eval hourly parquet chunks")
    parser.add_argument("--daily-path", default="data/daily_dataset/daily_df_eval.parquet",
                        help="Where to write/read eval daily aggregated+imputed parquet")
    parser.add_argument("--modelready-path", default="data/daily_dataset/daily_df_eval_modelready.parquet",
                        help="Where to write/read eval feature-ready parquet")
    # train feature-ready (already exists)
    parser.add_argument("--train-modelready-path", required=True,
                        help="Path to existing train feature-ready parquet")
    # categories and model settings
    parser.add_argument("--cats", nargs="+", type=int, required=True,
                        help="List of third_category_id to process")
    parser.add_argument("--model-dir", default="models",
                        help="Directory where nbeats_cat_{cat}.pt are saved")
    parser.add_argument("--input-len", type=int, default=28,
                        help="Number of historical days for model input")
    parser.add_argument("--output-len", type=int, default=7,
                        help="Number of days to forecast")
    args = parser.parse_args()

    # 1-3: prepare eval model-ready if missing
    if not os.path.exists(args.modelready_path):
        print("▶️ Step 1: stream & flatten (eval)")
        os.makedirs(args.flat_dir, exist_ok=True)
        stream_and_flatten(split="eval", batch_size=args.batch_size, output_dir=args.flat_dir)

        print("▶️ Step 2: aggregate & impute (eval)")
        aggregate_and_impute(input_dir=args.flat_dir, output_path=args.daily_path)

        print("▶️ Step 3: featurize (eval)")
        build_features(input_path=args.daily_path, output_path=args.modelready_path)
    else:
        print("▶️ Using existing eval feature-ready file")

    # 4: load train & eval feature tables
    df_train = pd.read_parquet(args.train_modelready_path)
    df_eval  = pd.read_parquet(args.modelready_path)

    # prepare montage subplots
    cats = args.cats
    n = len(cats)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*6, rows*4), sharex=True, sharey=True)
    axes = axes.flatten()

    # 5: loop categories, forecast and plot
    for i, cat in enumerate(args.cats):
        print(f"\n🔍 Processing category {cat}")

        # 5a) aggregate train series
        df_tr = (
            df_train[df_train["third_category_id"] == cat]
            .groupby("dt", as_index=False)["daily_sale_imputed"]
            .sum()
            .sort_values("dt")
        )
        # <— CAST HERE
        df_tr["daily_sale_imputed"] = df_tr["daily_sale_imputed"].astype("float32")

        ts_tr = TimeSeries.from_dataframe(
            df_tr, time_col="dt", value_cols="daily_sale_imputed", freq="D"
        )
        history = ts_tr[-args.input_len :]

        # 5b) aggregate eval series
        df_ev = (
            df_eval[df_eval["third_category_id"] == cat]
            .groupby("dt", as_index=False)["daily_sale_imputed"]
            .sum()
            .sort_values("dt")
        )
        # <— CAST HERE
        df_ev["daily_sale_imputed"] = df_ev["daily_sale_imputed"].astype("float32")

        ts_ev = TimeSeries.from_dataframe(
            df_ev, time_col="dt", value_cols="daily_sale_imputed", freq="D"
        )

        # 5b) aggregate eval series
        df_ev = (
            df_eval[df_eval["third_category_id"] == cat]
            .groupby("dt", as_index=False)["daily_sale_imputed"]
            .sum()
            .sort_values("dt")
        )
        ts_ev = TimeSeries.from_dataframe(
            df_ev, time_col="dt", value_cols="daily_sale_imputed", freq="D"
        )

        # 5c) load trained model
        model_path = os.path.join(args.model_dir, f"nbeats_cat_{cat}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # re‐instantiate the model shell with identical architecture & pl kwargs
        model = NBEATSModel(
            input_chunk_length=args.input_len,
            output_chunk_length=args.output_len,
            random_state=42,
            n_epochs=50,               # number of training epochs
            dropout=0.1,               # dropout in each layer
            batch_size=8,             # minibatch size
            pl_trainer_kwargs={}   # or whatever device settings you need
        )

        # load the saved weights into the Darts model
        model.load_weights(
            model_path,
            load_encoders=False,
            skip_checks=True,
            map_location="cpu"
        )

        # set to eval mode
        # model.eval()

        ts_tr = ts_tr.astype(np.float32)
        ts_ev = ts_ev.astype(np.float32)

        # 5d) forecast next output_len days
        pred_ts = model.predict(n=args.output_len, series=history)

        # 5e) merge predictions with actual eval
        df_pred = pred_ts.to_dataframe().reset_index().rename(
            columns={"index": "dt", "daily_sale_imputed": "prediction"}
        )
        df_cmp = pd.merge(df_pred, df_ev, on="dt", how="left").rename(
            columns={"daily_sale_imputed": "actual"}
        )

        # 5f) plot on montage
        ax = axes[i]  # use the index i from enumerating cats
        ax.plot(df_cmp["dt"], df_cmp["actual"], label="Actual")
        ax.plot(df_cmp["dt"], df_cmp["prediction"], label="Predicted")
        ax.set_title(f"Category {cat}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Sale")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

    # hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    # fig.tight_layout()
    plt.savefig("src/models/category_forecasts.png", bbox_inches="tight", dpi=150)

if __name__ == "__main__":
    main()