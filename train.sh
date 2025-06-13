#!/bin/bash

# 에러 발생 시 중단
set -e

# 학습 시작 시간 기록
start_time=$(date +"%Y-%m-%d %H:%M:%S")

models=("MF-Transformer" "MUSE" "LARP" "PISA")


for model in "${models[@]}"
do
    echo "================================================="
    echo "Starting training for $model at $(date +"%Y-%m-%d %H:%M:%S")"
    CUDA_VISIBLE_DEVICES=3 python src/main1.py --model_name "$model"
    echo "Completed training for $model at $(date +"%Y-%m-%d %H:%M:%S")"
    echo "================================================="
done

echo "All training completed!"
echo "Training started at: $start_time"
echo "Training finished at: $(date +"%Y-%m-%d %H:%M:%S")"