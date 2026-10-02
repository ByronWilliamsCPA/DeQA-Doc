#!/bin/bash
# DIQA-5000_1 Tier 1 LoRA fine-tuning
# Dataset: 13,163 samples (3,500 base + 9,663 expansion from 3 streams)
# Requires: 2x RTX 3090 or equivalent
#
# Usage:
#   LOAD=/path/to/mplug-owl2-base OUTPUT=./checkpoints/tier1 sh scripts/train_tier1_lora.sh

export PYTHONPATH=./:$PYTHONPATH

LOAD="${LOAD:?Set LOAD to mPLUG-Owl2 base model path}"
OUTPUT="${OUTPUT:-./checkpoints/diqa_tier1_lora}"

deepspeed --include localhost:0,1 --master_port 6688 src/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "$LOAD" \
    --version v1 \
    --lora_enable True \
    --dataset_type single \
    --level_prefix "The quality of the image is" \
    --level_names excellent good fair poor bad \
    --softkl_loss True \
    --weight_rank 0.0 \
    --weight_softkl 1.0 \
    --weight_next_token 0.005 \
    --continuous_rating_loss True \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths Data-DeQA-Score/DIQA-5000_1/train_overall.json \
    --data_weights 1 \
    --image_folder Data-DeQA-Score \
    --output_dir "$OUTPUT" \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
