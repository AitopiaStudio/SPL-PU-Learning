#!/bin/bash
# scripts/run_exp.sh
# Run on the included Fair For You credit dataset (data/sample/)
# This is the dataset used in the UCL dissertation — ready to run out of the box!

python main_2.py \
    --dataset exp \
    --data_dir data/sample/ \
    --prior 0.2 \
    --no_cuda \
    --pre_loss bce \
    --loss bce \
    --spl_type linear \
    --scheduler_type exp \
    --alpha 0.1 \
    --eta 1.1 \
    --max_thresh 2.0 \
    --grow_steps 10 \
    --epochs 100 \
    --pre_epochs 400 \
    --pre_lr 0.001 \
    --lr 0.0001 \
    --batch_size 64 \
    --batch_size_val 128 \
    --patience 5 \
    --seed 0
