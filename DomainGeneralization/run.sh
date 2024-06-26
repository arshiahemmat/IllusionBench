#!/bin/bash

# Define the algorithms you want to iterate over
algorithms=("CLIPERM" "CLIPIRM"  "CLIPMixUp" "CLIPRegMixUp" "CLIPVREx" "CLIPDRO")

for algorithm in "${algorithms[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.new_train \
        --data_dir ./path/to/data \
        --output_dir ./result \
        --algorithm $algorithm \
        --dataset SIN \
        --hparams '{"backbone": "resnet50", "clip_backbone": "RN50"}'
done