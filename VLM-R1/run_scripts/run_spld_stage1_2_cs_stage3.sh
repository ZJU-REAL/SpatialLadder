PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}" # TODO: change this to your own 
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="${REPO_HOME}/data/spld_spatial_data.jsonl" 
image_folders="${REPO_HOME}/data/images"
model_path=${REPO_HOME}/checkpoints/spld_stage1_2_cs
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="spld_stage1_2_cs_stage3" # TODO: change this to your own experiment name
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
# create the run directory and log file
mkdir -p ${REPO_HOME}/runs/${EXP_NAME}/log
export LOG_PATH="${REPO_HOME}/runs/${EXP_NAME}/log/debug_log.$(date +%Y-%m-%d-%H-%M-%S).txt"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export WANDB_MODE=disabled

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo_spld_stage3.py \
    --use_vllm False \
    --output_dir ${REPO_HOME}/checkpoints/${EXP_NAME} \
    --resume_from_checkpoint true \
    --model_name_or_path $model_path \
    --data_file_paths $data_paths \
    --image_folders $image_folders \
    --is_reward_customized_from_vlm_module $is_reward_customized_from_vlm_module \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --num_train_epochs 0.0001 \
    --bf16 \
    --attn_implementation flash_attention_2 \
    --run_name ${EXP_NAME} \
    --save_steps 200 \
    --num_generations 4 \
    --max_completion_length 1024 \
    --beta 0.01 \
    --report_to wandb \
    --dataset-name not_used \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --reward_funcs accuracy format \
    --accuracy_reward_weight 1 \
    --format_reward_weight 1 \
    --max_pixels $((128 * 28 * 28)) \
    --min_pixels $((16 * 28 * 28)) \

echo "Training completed for ${EXP_NAME}"