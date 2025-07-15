# train_darts_nbeats.py
"""
Train and evaluate Darts N-BEATS on top third-level categories.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.metrics import mae
from darts.models import NBEATSModel

def main(args):
    df = pd.read_parquet(args.input_path)
    df["dt"] = pd.to_datetime(df["dt"])
    # pick categories
    if args.categories:
        cats = args.categories
    else:
        # fallback: top-n by total sales
        sales = df.groupby("third_category_id")["daily_sale_imputed"].sum()
        cats = sales.nlargest(args.top_n).index.tolist()
    os.makedirs(args.output_dir, exist_ok=True)
    summary = []

    for cat in cats:
        df_cat = df[df["third_category_id"] == cat]
        agg = (
            df_cat.groupby("dt", as_index=False)["daily_sale_imputed"]
                  .sum()
        )
        series = TimeSeries.from_dataframe(
            agg, time_col="dt", value_cols="daily_sale_imputed",
            fill_missing_dates=True, freq="D"
        )
        train, val = series.split_after(0.8)

        model = NBEATSModel(
            input_chunk_length=args.input_chunk_length,
            output_chunk_length=args.output_chunk_length,
            n_epochs=args.n_epochs,
            dropout=args.dropout,
            batch_size=args.batch_size,
            random_state=42,
            force_reset=True
        )
        model.fit(train, verbose=False)
        pred = model.predict(len(val), series=train)
        error = mae(val, pred)
        print(f"Cat {cat} → MAE: {error:.2f}")

        # plot & save
        plt.figure(figsize=(10,4))
        val.plot(label="actual")
        pred.plot(label="forecast")
        plt.title(f"Category {cat} Forecast")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/cat_{cat}_forecast.png")
        plt.close()

        # save model
        model.save(f"{args.output_dir}/model_cat_{cat}.pth.tar")

        summary.append({"category": cat, "mae": error})

    pd.DataFrame(summary).to_csv(f"{args.output_dir}/mae_summary.csv", index=False)
    print("✅ All done — summary saved to mae_summary.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path",       type=str,
                        default="data/daily_dataset/daily_df_modelready.parquet")
    parser.add_argument("--output-dir",       type=str,
                        default="models/nbeats")
    parser.add_argument("--categories",       type=int, nargs="*",
                        help="Specific third_category_ids to train on")
    parser.add_argument("--top-n",            type=int, default=3,
                        help="Pick top-n categories by sales if --categories unset")
    parser.add_argument("--input-chunk-length", type=int, default=28)
    parser.add_argument("--output-chunk-length", type=int, default=7)
    parser.add_argument("--n-epochs",         type=int, default=50)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--batch-size",       type=int, default=32)
    args = parser.parse_args()
    main(args)