# Notebooks Overview

### 🔹 01_eda.ipynb

Objective: Perform foundational exploratory data analysis on the full FreshRetailNet dataset.
Key Steps:
- Identified unique products (865), stores (898), and ~50,000 product-store combinations.
- Plotted category and store distributions to assess modeling feasibility.
- Helped guide whether to model per-product/store or via aggregation.

### 🔹 02_category_store_analysis.ipynb

Objective: Analyze demand distribution across categories and stores to design scalable modeling groups.
Key Steps:
- Identified top 20 third-level categories globally and top 5 per store.
- Found 87 unique categories cover 83.1% of demand.
- Mapped each of these top categories to associated store IDs for selective model training.

### 🔹 03_latent_demand_forecasting.ipynb

Objective: Set up demand forecasting pipelines focused on the 87 high-impact third-level categories.
Key Steps:
- Filtered dataset to only include relevant categories.
- Calculated total demand coverage.
- Prepared modeling granularity plan to balance performance with scalability.

### 🔹 04_product_level_demand_imputation.ipynb

Objective: Impute missing or latent demand signals at the product level prior to forecasting.
Key Steps:
- Designed a strategy to estimate imputed demand using hours-level stock/sales signals.
- Merged and aligned imputed values with the master dataset.
- Enabled cleaner downstream modeling by reducing signal sparsity.
  
### 🔹 05_daily_baseline_modeling.ipynb

Objective: Aggregate hourly sales data into daily format and prepare a baseline dataset for modeling.
Key Steps:
- Aggregated hourly features (sales, stockouts, weather, promotions) to daily granularity.
- Flagged full-day stockout periods (`oos_hours_24 == 24`) and suspicious sales during those periods.
- Switched from 6–22 hour filtering to full 24-hour retention due to inconsistencies in industrial reporting.
- Finalized `daily_df` for top third-level categories covering 90–95% of total sales volume.


### 🔹 06_model_training_analysis.ipynb

Objective: Analyze baseline model performance and refine training approach.
Key Steps:
- Evaluated LightGBM daily‐baseline RMSE/MAE per category.
- Reviewed feature importance and correlation diagnostics (ACF/PACF).
- Logged model stability issues and prepared for recursive/direct strategies.

### 🔹 07_imputation_and_aggregation.ipynb

Objective: Implement scalable imputation and aggregation pipeline for hourly→daily transformation.
Key Steps:
- Chunked read of flattened hourly parquet files.
- Raw and in‐stock group aggregations with vectorized imputation.
- Exported final daily dataset with imputed sales.

### 🔹 08_feature_engineering.ipynb

Objective: Build comprehensive time series features for modeling.
Key Steps:
- Generated multiple lags (1,7,14 days) and rolling statistics.
- Added calendar encodings (day_of_week, weekend, time_idx).
- Integrated contextual features (stockouts, weather, promotions).

### 🔹 08_model_recursive.ipynb

Objective: Train recursive autoregressive LightGBM models per category using skforecast.
Key Steps:
- Prepared per‐category train/validation splits.
- Fitted `ForecasterRecursive` with lag features and exogenous variables.
- Reported per‐category RMSE and compared against baseline.

### 🔹 09_direct_sliding_window.ipynb

Objective: Establish direct multi‐step sliding‐window forecasting baseline.
Key Steps:
- Constructed fixed‐length lag windows as features.
- Trained multi‐output regressors for 7‐day forecasts.
- Benchmarked against recursive and baseline models.

### 🔹 10_sequence_modeling.ipynb

Objective: Prototype sequence‐to‐sequence forecasting in pure PyTorch.
Key Steps:
- Implemented encoder‐decoder architectures.
- Built custom `DataLoader` and training loops.
- Assessed initial performance and GPU feasibility.

### 🔹 11_Sequence_Modelling_GPU.ipynb

Objective: Accelerate sequence modeling with PyTorch Forecasting (TFT).
Key Steps:
- Converted daily data into `TimeSeriesDataSet`.
- Trained Temporal Fusion Transformer on MPS/CUDA.
- Logged missing‐timestep handling, model convergence, and attention plots.

### 🔹 12-darts-n-beats.ipynb

Objective: Experiment with Darts N‑BEATS forecasting at category level.
Key Steps:
- Aggregated daily category series and filled missing dates.
- Configured N‑BEATSModel with cyclic encoders.
- Ran backtests and plotted predicted vs actual sales.

### 🔹 14-darts-n-beats.ipynb

Objective: Follow-up on N‑BEATS hyperparameter and encoder studies.
Key Steps:
- Tuned block/types, stack_depth, and encoder settings.
- Evaluated additional temporal encodings (datetime attributes).
- Compared performance against baseline regressors.