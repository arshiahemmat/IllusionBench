#!/bin/bash

# Define the algorithms you want to iterate over
algorithms=("CLIPERM" "CLIPIRM"  "CLIPMixUp" "CLIPRegMixUp" "CLIPVREx" "CLIPDRO")
# algorithms=("CLIP" "DPLCLIP")
# 

for algorithm in "${algorithms[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python -m domainbed.scripts.new_train \
        --data_dir /homes/55/jianhaoy/projects/SIN \
        --output_dir /homes/55/jianhaoy/projects/DPLCLIP/res \
        --algorithm $algorithm \
        --dataset SIN \
        --hparams '{"backbone": "resnet50", "clip_backbone": "RN50"}'
done