############################################################
# Dockerfile for FreshRetail Demand Forecasting Pipeline
############################################################

# Base image with Python 3.12
FROM python:3.12-slim

# Set working directory
WORKDIR /app

#  Paths are intended to be mounted as volumes
VOLUME ["/app/data", "/app/models"]

# Pull in GNU OpenMP so LightGBM (and thus Darts) can load its C extension
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Default command (you can override this to train or predict)
CMD ["bash", "-c", "echo 'Please pass a command to run training or prediction' && exit 1"]

# ---------------------------------------------
# Example Docker run commands:
# 1) Training:
#    docker run --rm freshretail-forecast:latest \
    #  python src/train_pipeline.py \
    #    --split train \
    #    --batch-size 12000 \
    #    --flat-dir data/flattened_chunks \
    #    --daily-path data/daily_dataset/daily_df_imputed.parquet \
    #    --modelready-path data/daily_dataset/daily_df_modelready.parquet \
    #    --cats 81 60 82 184 1 \
    #    --model-dir models \
    #    --input-len 28 \
    #    --output-len 7
#
# 2) Prediction:
#    docker run --rm freshretail-forecast:latest \
#      python src/prediction_pipeline.py \
#        --train-modelready-path data/daily_dataset/daily_df_modelready.parquet \
#        --flat-dir data/flattened_chunks_eval \
#        --daily-path data/daily_dataset/daily_df_eval.parquet \
#        --modelready-path data/daily_dataset/daily_df_eval_modelready.parquet \
#        --cats 81 60 82 184 1 \
#        --model-dir models \
#        --input-len 28 \
#        --output-len 7
#
# To run:
#  - Use a single Docker image (freshretail-forecast:latest).
#  - Invoke one container instance per task (train or predict).
#  - No need for multiple images—just override the CMD.
# ---------------------------------------------
