# Suppress pkg_resources deprecation warning
import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning
)

# Suppress MPS pin_memory warning
import torch
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now.*",
    category=UserWarning
)

import os
import argparse
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models.forecasting.nbeats import NBEATSModel
from sklearn.metrics import mean_absolute_error
import torch
from darts.models.forecasting.nbeats import NBEATSModel

def save_nbeats_model(model, path):
    """
    Saves only the state_dict of model.model to the given path.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # pull out the raw Lightning module inside Darts
    lightning_module = model.model  

    # save only its weights
    torch.save(lightning_module.state_dict(), path)
    model.save(path, clean=True)

def load_nbeats_model(
    path: str,
    input_chunk_length: int,
    output_chunk_length: int,
    random_state: int,
    pl_trainer_kwargs: dict
) -> NBEATSModel:
    """
    Instantiates a fresh NBEATSModel and loads the saved state_dict.
    Returns the ready-to-use model.
    """
   # Re-create the same wrapper
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        random_state=random_state,
        pl_trainer_kwargs=pl_trainer_kwargs
    )
    # Load *only* weights (and encoders)
    weights_path = path
    model.load_weights(
        weights_path,
        load_encoders=False,
        skip_checks=True,
        map_location="cpu"
    )
    return model


def train_for_category(
    df: pd.DataFrame,
    cat_id: int,
    input_len: int,
    output_len: int,
    model_dir: str
):
    # 1) build a daily series for this category
    df_cat = (
        df[df["third_category_id"] == cat_id]
        .groupby("dt", as_index=False)["daily_sale_imputed"]
        .sum()
        .sort_values("dt")
    )
    ts = TimeSeries.from_dataframe(
        df_cat, time_col="dt", value_cols="daily_sale_imputed", freq="D"
    )
    # cast series to float32 to match trainer precision
    ts = ts.astype(np.float32)

    # 2) hold out exactly the last `output_len` days
    train, val = ts[:-output_len], ts[-output_len:]

    # 3) instantiate & fit
    # configure PyTorch Lightning trainer parameters
    pl_kwargs = {"precision": 32}
    if torch.backends.mps.is_available():
        pl_kwargs.update({"accelerator": "mps", "devices": 1})

    model = NBEATSModel(
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        random_state=42,
        n_epochs=50,               # number of training epochs
        dropout=0.1,               # dropout in each layer
        batch_size=8,             # minibatch size
        pl_trainer_kwargs=pl_kwargs
    )
    model.fit(train, verbose=False)
    # save weights via helper
    weights_path = os.path.join(model_dir, f"nbeats_cat_{cat_id}.pt")
    save_nbeats_model(model, weights_path)

    # 4) predict + eval
    preds = model.predict(n=output_len)
    true = val.values()[:, 0]
    pred = preds.values()[:, 0]
    mae  = mean_absolute_error(true, pred)
    print(f"✔️  Cat {cat_id}: MAE={mae:.2f}")

    # # 5) save weights only (state_dict)
    # os.makedirs(model_dir, exist_ok=True)
    # state_path = os.path.join(model_dir, f"nbeats_cat_{cat_id}.pth")
    # torch.save(model.model.state_dict(), state_path)

    return mae


def predict_for_category(
    category: int,
    modelready_path: str,
    model_dir: str,
    input_len: int = 28,
    output_len: int = 7
) -> pd.DataFrame:
    import os
    import pandas as pd
    import numpy as np
    from darts import TimeSeries

    # 1) load features
    if not os.path.isfile(modelready_path):
        raise FileNotFoundError(f"No feature-ready parquet at {modelready_path}")
    df = pd.read_parquet(modelready_path)

    # 2) filter & aggregate daily
    df_cat = (
        df[df["third_category_id"] == category]
        .groupby("dt", as_index=False)["daily_sale_imputed"]
        .sum()
        .sort_values("dt")
    )
    if df_cat.empty:
        raise ValueError(f"No data for category {category}")

    # 3) build series
    ts = TimeSeries.from_dataframe(df_cat, time_col="dt", value_cols="daily_sale_imputed", freq="D")
    ts = ts.astype(np.float32)
    print("Training time series:", ts)
    # load model via helper
    pl_kwargs = {"precision": 32}
    if torch.backends.mps.is_available():
        pl_kwargs.update({"accelerator": "mps", "devices": 1})
    weights_path = os.path.join(model_dir, f"nbeats_cat_{category}.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"No weights file at {weights_path}")
    model = load_nbeats_model(
        weights_path,
        input_chunk_length=input_len,
        output_chunk_length=output_len,
        random_state=42,
        pl_trainer_kwargs=pl_kwargs
    )

    # 5) sliding-window backtest via historical_forecasts
    backtest_series = model.historical_forecasts(
        series=ts,
        forecast_horizon=output_len,
        stride=output_len,
        retrain=False,
        last_points_only=False,
        verbose=True
    )
    # concatenate all forecast segments into one TimeSeries
    pred_ts = backtest_series[0]
    for segment in backtest_series[1:]:
        pred_ts = pred_ts.append(segment)

    # 6) to DataFrame
    df_pred = pred_ts.pd_dataframe().reset_index()
    df_pred.columns = ["dt", "prediction"]
    df_pred["third_category_id"] = category
    return df_pred


def main(
    cats: list[int],
    modelready_path: str,
    model_dir: str,
    input_len: int = 28,
    output_len: int = 7
):
    df = pd.read_parquet(modelready_path)
    print(f"ℹ️  Loaded model-ready data: {df.shape[0]} rows")
    for cat in cats:
        train_for_category(df, cat, input_len=input_len, output_len=output_len, model_dir=model_dir)
    print("✅ All categories trained.")