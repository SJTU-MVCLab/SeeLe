#!/bin/bash

models=("seele")

for model in "${models[@]}"; do
    bash run_train.sh $model
    bash run_generate_cluster.sh $model
    bash run_finetune.sh $model
    bash run_mask_render.sh $model
done