#!/bin/bash

# Stop immediately if any command fails.
set -e

# --- Accelerate settings ---
# Number of GPUs to use (e.g., 4)
NUM_GPUS=2
# GPU ID list to use (e.g., 0, 1, 2, 3)
GPU_IDS="1,0"
# Optional Accelerate config file path ('accelerate config' can generate one)
# ACCELERATE_CONFIG_FILE="/path/to/your/accelerate_config.yaml"
# Main process port for distributed communication (default 29500, change if conflict)
MAIN_PROCESS_PORT=29500

# --- Load environment variables ---
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
else
    echo "Missing .env file with WANDB settings."
    exit 1
fi

# --- Training script args ---
export WANDB_BASE_URL
export WANDB_API_KEY
export WANDB_MODE
wandb login


# Enable detailed PyTorch distributed debug logs.
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --- Experiment list ---
#EXP_LIST=("exp0" "exp1" "exp2")
EXP_LIST=("default")

# --- Run accelerate launch ---
for exp_name_item in "${EXP_LIST[@]}"
do
    echo "Running experiment: $exp_name_item"
    accelerate launch \
        --num_processes $NUM_GPUS \
        --gpu_ids $GPU_IDS \
        --main_process_port $MAIN_PROCESS_PORT \
        train.py \
        --config_path "exp_config/config.py" \
        --config_name "Config" \
        --resume_from_checkpoint "" \
        --exp_name "$exp_name_item" 
done
echo "Training script finished."
