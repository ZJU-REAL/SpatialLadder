PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
export REPO_HOME="${PROJECT_ROOT}" # TODO: change this to your own 
echo "REPO_HOME: $REPO_HOME"
# on remote
data_paths="${REPO_HOME}/data/spld_coldstart_data.jsonl" 
image_folders="${REPO_HOME}/data/images"
model_path=${REPO_HOME}/checkpoints/spld_stage1_2
is_reward_customized_from_vlm_module=False
echo "data_paths: $data_paths"
echo "image_folders: $image_folders"

export EXP_NAME="spld_stage1_2_cs" # TODO: change this to your own experiment name
cd ${REPO_HOME}/src/open-r1-multimodal

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export WANDB_MODE=disabled

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/sft_spld_cs.py \
    --output_dir ${REPO_HOME}/checkpoints/${EXP_NAME} \
    --model_name_or_path $model_path \
    --dataset_name $data_paths \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --deepspeed ${REPO_HOME}/src/open-r1-multimodal/local_scripts/zero3.json \
    --num_train_epochs 0.001 \
    --run_name spld_stage1_2_cs \
    --save_steps 200 \
    --save_only_model true \
    --image_folders $image_folders \