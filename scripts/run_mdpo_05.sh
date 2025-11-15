#!/bin/bash
# echo "start"
# sleep 430m
# echo "End"

task_name=llava15_7b_DPO
exp_name=llava15_7b_mdpo_05

deepspeed --master_port 25433 --include  localhost:4,5,6,7\
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path /NAS/fengjw/.cache/base_models/llava-v1.5-7b \
    --data_path /NAS/fengjw/.cache/base_datasets/new_test_logps \
    --image_folder not_used \
    --vision_tower /NAS/fengjw/.cache/base_models/vision_tower-clip336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /NAS/fengjw/.cache/output/DAMA/llava7b_mdpo_model_05 \
    --num_train_epochs 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 20 \
    --learning_rate 5e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_dir /NAS/fengjw/.ckpt/RePO/$task_name-$exp_name/log \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --task DPO \
    --report_to wandb \
    --run_name $exp_name \
    --dataloader_num_workers 0 \
    --dpo_use_average False \
    --dpo_token_weighted False \
    --dpo_token_weight 1.0 \
    --dpo_beta 0.1
