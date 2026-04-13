#!/bin/bash
# scripts/run_risk.sh
# Run on Home Credit Default Risk dataset
# Requires: data/risk/application_train.csv and data/risk/application_test.csv
# Download from: https://www.kaggle.com/c/home-credit-default-risk

python main_2.py \
    --dataset risk \
    --prior 0.2 \
    --no_cuda \
    --data_dir data/ \
    --pre_loss bce \
    --loss bce \
    --spl_type linear \
    --scheduler_type exp \
    --alpha 0.2 \
    --eta 1.1 \
    --epochs 100 \
    --pre_epochs 400 \
    --lr 0.0001 \
    --batch_size 64 \
    --patience 5 \
    --seed 0
