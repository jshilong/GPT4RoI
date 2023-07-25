#!/bin/bash

WORKDIR=${1:-./exp/stage1}
mkdir -p $WORKDIR

export PYTHONPATH=`pwd`:$PYTHONPATH
# only train the spi module
export ONLY_SPI=1


torchrun --nnodes=1 --nproc_per_node=4 --master_port=25002 \
    gpt4roi/train/train_mem.py \
    --model_name_or_path path_to_vicuna-7b  \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    --dataset_config ./gpt4roi/configs/stage1.py \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir $WORKDIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "none" \
    --seed 0 \
    | tee $WORKDIR/train.log