#!/bin/bash

# Configuration
APP_NAME="m1_rf_sub001_dynamic_2c13exec16GB"
MASTER="yarn"
DEPLOY_MODE="cluster"
SCRIPT_PATH="/home/efs/erasmus/e2405193/ubs_bigdata/UBS-BigData-24/m1/scripts/m1_train_vlocal_transformers.py"
SUBSAMPLE="0.01"
NUM_EXECUTORS="13"
EXECUTOR_CORES="2"
EXECUTOR_MEMORY="16g"
DRIVER_MEMORY="16g"
LOGFILE_NAME="m1_rf_sub001_dynamic_2c13exec16GB.log"
S3_BUCKET="s3://ubs-cde/home/e2405193/bigdata/log_files"

# Run Spark-submit with nohup and log output to a file
nohup spark-submit \
    --name "$APP_NAME" \
    --master "$MASTER" \
    --deploy-mode "$DEPLOY_MODE" \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.minExecutors=2 \
    --conf spark.dynamicAllocation.maxExecutors="$NUM_EXECUTORS" \
    --conf spark.dynamicAllocation.initialExecutors=2 \
    --conf spark.shuffle.service.enabled=true \
    --conf spark.akka.logLevel=INFO \
    --executor-cores "$EXECUTOR_CORES" \
    --executor-memory "$EXECUTOR_MEMORY" \
    --driver-memory "$DRIVER_MEMORY" \
    "$SCRIPT_PATH" \
    --subsample "$SUBSAMPLE" \
    > "$LOGFILE_NAME" 2>&1 &

# Upload logfile to S3
nohup aws s3 cp "$LOGFILE_NAME" "$S3_BUCKET/$LOGFILE_NAME" &

# Dynamic allocation configurations (commented out)
#     --num-executors "$NUM_EXECUTORS" \
