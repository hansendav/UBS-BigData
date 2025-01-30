#!/bin/bash

META_PATH='s3://ubs-cde/home/e2405193/bigdata/meta_with_image_paths.parquet'
TF_CONFIG_PATH='./tf_config_w1.json' # check this here 
WORKER_ID='1'
BATCHSIZE='8'
NUM_WORKERS='4'
CHKPT_NAME='4NODES_1PERC_NEWMETHOD_SHARD'
LOG_NAME="WORKER${WORKER_ID}_4NODES_1PERC_NEWMETHOD_SHARD" 
LOG_FILE_NAME="$LOG_NAME.log"
SUBSAMPLE='0.03'

# Execute the Python script with the provided arguments
nohup python tf_distributed_training_dataing.py \
    --meta_path $META_PATH \
    --tfconfig_path $TF_CONFIG_PATH \
    --batchsize $BATCHSIZE \
    --num_workers $NUM_WORKERS \
    --chkpt_name $CHKPT_NAME \
    --log_name $LOG_NAME \
    --subsample $SUBSAMPLE \
    > $LOG_FILE_NAME 2>&1 &

nohup aws s3 cp $LOG_FILE_NAME s3://ubs-cde/home/e2405193/bigdata/experiments/m2/logs/stdout/$LOG_FILE_NAME &