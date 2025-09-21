#!/bin/bash

# 显存占用阈值（MB），低于此值认为GPU空闲
MEMORY_THRESHOLD=200  # 默认200MB，可调整

# 日志函数（带时间戳）
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查GPU显存占用是否低于阈值
wait_for_gpu_memory_idle() {
    while true; do
        # 使用nvidia-smi获取显存占用（MB）
        memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        
        # 检查所有GPU显存占用是否都低于阈值
        all_idle=true
        for mem in $memory_used; do
            if [ "$mem" -ge "$MEMORY_THRESHOLD" ]; then
                all_idle=false
                break
            fi
        done

        if $all_idle; then
            log "All GPUs have memory usage below ${MEMORY_THRESHOLD}MB. Starting program..."
            break
        else
            log "GPU memory is in use (${memory_used} MB >= ${MEMORY_THRESHOLD} MB). Waiting..."
            sleep 30  # 每30秒检查一次
        fi
    done
}

# 主程序
log "===== Script started ====="
log "Checking GPU memory usage (threshold: ${MEMORY_THRESHOLD} MB)..."
wait_for_gpu_memory_idle


deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/finetune_for_reasoning.json \
    --image_folder ./playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-reasoning \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
