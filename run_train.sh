#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1

accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --mixed_precision bf16 \
    train.py
