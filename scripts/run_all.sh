#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
dataset_base_path="dataset/seele" # PATH TO YOUR DATASET
output_base_path="output/seele" # PATH TO YOUR OUTPUT

datasets=("counter") # Replace with your actual dataset names
# datasets=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill" "train" "truck" "playroom" "drjohnson")

for dataset in "${datasets[@]}"; do
    model_path="$output_base_path/$dataset"
    dataset_path="$dataset_base_path/$dataset"

    echo "Train dataset: $dataset"
    python3 train.py -m $model_path -s $dataset_path --eval

    echo "Generate clusters for dataset: $dataset"
    if [[ "$dataset" == "playroom" || "$dataset" == "drjohnson" ]]; then
        python3 generate_cluster.py -m $model_path -n 8
    else
        python3 generate_cluster.py -m $model_path -n 4
    fi

    echo "Finetune dataset: $dataset"
    python3 finetune.py \
        -s $dataset_path \
        -m $model_path \
        --start_checkpoint "$model_path/chkpnt30000.pth" \
        --eval \
        --iterations 31_000     

    echo "Render dataset: $dataset"
    python3 seele_render.py -m $model_path -s $dataset_path --eval --load_finetune --save_image --debug
    
    echo "Metrics for dataset: $dataset"
    python3 metrics.py -m $model_path
done
