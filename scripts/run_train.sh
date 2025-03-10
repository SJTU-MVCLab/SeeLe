#!/bin/bash

# List of dataset names
# datasets=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill" "train" "truck" "playroom" "drjohnson")
datasets=("counter") # Replace with your actual dataset names

# Path to models
model_base_path="output/$1" # PATH TO YOUR MODELS

dataset_base_path="dataset/$1" # PATH TO YOUR DATASET

# Iterate over each dataset
for dataset_name in "${datasets[@]}"; do
    echo "Processing dataset: $dataset_name\n"
    python3 train.py -m "$model_base_path/$dataset_name" -s "$dataset_base_path/$dataset_name" --eval
    echo "Test:\n"
    python3 render.py -m "$model_base_path/$dataset_name" -s "$dataset_base_path/$dataset_name" --skip_train --eval 
    python3 metrics.py -m "$model_base_path/$dataset_name" 
done

# Completion signal
echo "All datasets processed. Task complete."