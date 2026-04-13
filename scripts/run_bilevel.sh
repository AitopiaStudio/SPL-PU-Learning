#!/bin/bash
# scripts/run_bilevel.sh
# Run with bilevel hyperparameter optimization (CG method)

python main.py \
    --dataset mnist \
    --prior 0.2 \
    --bilevel \
    --outer_iters 5 \
    --outer_lr 0.01 \
    --hyper_K 10 \
    --epochs 50 \
    --pre_epochs 400 \
    --pre_loss nnpu \
    --loss bce \
    --batch_size 64 \
    --lr 0.001 \
    --patience 10 \
    --seed 0
