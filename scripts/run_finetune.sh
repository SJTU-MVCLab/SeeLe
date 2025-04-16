#!/bin/bash
# Function to get an available GPU with memory usage below the threshold
get_available_gpu() {
  local mem_threshold=25000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
    awk -v threshold="$mem_threshold" -F', ' '
      $2 < threshold { print $1; exit }
    '
}

# List of dataset names
# datasets=("bicycle" "bonsai" "counter" "flowers" "garden" "kitchen" "room" "stump" "treehill" "train" "truck" "playroom" "drjohnson")
datasets=("counter") # Replace with your actual dataset names

# Path to models
model_base_path="output/seele" # PATH TO YOUR MODELS

dataset_base_path="dataset/seele" # PATH TO YOUR DATASET
port=6035

# Iterate over each dataset
for dataset_name in "${datasets[@]}"; do
    echo "Processing dataset: $dataset_name"
    
    # Find an available GPU
    while true; do
        available_gpu=$(get_available_gpu)
        if [ -z "$available_gpu" ]; then
            echo "No GPU available with memory usage below threshold. Waiting..."
            sleep 60
            continue
        fi

        echo "Using GPU: $available_gpu"
        # Run the Python script with the selected GPU
        CUDA_VISIBLE_DEVICES="$available_gpu" python finetune.py \
         -s "$dataset_base_path/$dataset_name" \
         -m "$model_base_path/$dataset_name" \
         --start_checkpoint "$model_base_path/$dataset_name/chkpnt30000.pth" \
         --eval \
         --iterations 31_000   
        break
    done
done

# Wait for all background processes to finish
wait

# Completion signal
echo "All datasets processed. Task complete."