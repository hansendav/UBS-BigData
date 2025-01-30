#!/bin/bash

# SparkSession and script configurations
APP_NAME="prepare_cu_bigearthnetv2_pqfile"
MASTER="yarn"
DEPLOY_MODE="cluster" # spark driver on a cluster node! -> find stdout on the driver
SCRIPT_PATH="/home/efs/erasmus/e2405193/ubs_bigdata/UBS-BigData-24/m2/scripts/concatenate_patch_bands.py"
NUM_EXECUTORS="13"
EXECUTOR_CORES="2"
EXECUTOR_MEMORY="16g"
DRIVER_MEMORY="16g"
PARTITIONS="120"
LOGFILE_NAME="prepare_cu_bigearthnetv2_pqfile.log"
LOGFILE_PATH="s3://ubs-cde/home/e2405193/bigdata/m2/log_files"

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
    --partitions "$PARTITIONS" \
    > "$LOGFILE_NAME" 2>&1 &

# Upload logfile to S3
nohup aws s3 cp "$LOGFILE_NAME" "$LOGFILE_PATH/$LOGFILE_NAME" & 