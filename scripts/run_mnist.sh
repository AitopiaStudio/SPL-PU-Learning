#!/bin/bash
# scripts/run_mnist.sh
# Quick experiment on MNIST with SPL-PU

python main.py \
    --dataset mnist \
    --prior 0.2 \
    --pre_loss nnpu \
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
    --patience 5 \
    --seed 0
