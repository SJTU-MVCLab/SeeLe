#!/bin/bash

models=("seele")

for model in "${models[@]}"; do
    bash scripts/run_train.sh $model
    bash scripts/generate_cluster.sh $model
    bash scripts/run_finetune.sh $model
    bash scripts/run_seele_render.sh $model
done