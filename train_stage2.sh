export PYTHONPATH=`pwd`:$PYTHONPATH
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
    gpt4roi/train/train_mem.py \
    --model_name_or_path /mnt/petrelfs/share_data/zhangshilong/vicuna-7b/ \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter LLaVA-7b-pretrain-projector-v0-CC3M-595K-original_caption.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./exp/stage2 \
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
    | tee ./exp/stage2/train.log