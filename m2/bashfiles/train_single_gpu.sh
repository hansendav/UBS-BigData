#!/bin/bash

PATH_TO_PARQUET='s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet'
SUBSAMPLE_RATIO="0.03"
BATCH_SIZE='8'
CHECKPOINT_NAME='BASELINE_3PERC_NEWMETHOD'
LOGS_NAME='BASELINE_3PERC_NEWMETHOD'
LOG_FILE_NAME='BASELINE_3PERC_NEWMETHOD.log'

# Execute the Python script with the provided arguments

nohup python tf_single_gpu_train.py \
    --path_to_parquet $PATH_TO_PARQUET \
    --subsample_ratio $SUBSAMPLE_RATIO \
    --batch_size $BATCH_SIZE \
    --checkpoint_name $CHECKPOINT_NAME \
    --logs_name $LOGS_NAME \
    > $LOG_FILE_NAME 2>&1 &

nohup aws s3 cp $LOG_FILE_NAME s3://ubs-cde/home/e2405193/bigdata/experiments/m2/logfiles/$LOG_FILE_NAME &
