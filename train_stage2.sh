#!/bin/bash

WORKDIR=${1:-./exp/stage2}
STAGE1WORKDIR=${2:-./exp/stage1}
mkdir -p $WORKDIR



# Check if workdir is empty
if [ "$(find $WORKDIR -mindepth 1 -maxdepth 1 -type d)" ]; then
    echo "WORKDIR is not empty, resume training"
else
    # Check if STAGE1WORKDIR exists
    if [ ! -d $STAGE1WORKDIR ]; then
        echo "Error: Stage1 work directory $STAGE1WORKDIR does not exist."
        exit 1
    fi
    echo "WORKDIR is empty, load the model form stage1 workdir $STAGE1WORKDIR"
   # If empty, create checkpoint-0 directory and soft link all files from stage 1 except for 'scheduler.pt', 'training_args.bin', and 'optimizer.pt', so we can load the checkpoint from stage 1
    mkdir -p $WORKDIR/checkpoint-0
    cd $WORKDIR/checkpoint-0/
    find $STAGE1WORKDIR/* -type f -not -name 'scheduler.pt' -not -name 'training_args.bin' -not -name 'optimizer.pt' -not -name 'trainer_state.json' -print0 | xargs -0 -I {} ln -s {} .
    cd -
fi


export PYTHONPATH=`pwd`:$PYTHONPATH

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    gpt4roi/train/train_mem.py \
    --model_name_or_path path_to_vicuna-7b \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    --dataset_config ./gpt4roi/configs/stage2.py \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir $WORKDIR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.003 \
    --warmup_steps 3000 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to "none" \
    --seed 0 \
    | tee $WORKDIR/train.log