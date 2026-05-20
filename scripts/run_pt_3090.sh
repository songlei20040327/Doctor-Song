#!/bin/bash
# ============================================================================
# Doctor.Song 阶段1：领域持续预训练 (PT)
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：全参数训练 (use_peft=False)
# 产出：outputs-pt-doctor-song/
# ============================================================================

# ---- 网络配置：国内优先用镜像 ----
# 先试直连，不行自动切镜像
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null 2>&1; then
    echo "HF 直连可用"
else
    echo "HF 不可用，切换到镜像"
    export HF_ENDPOINT=https://hf-mirror.com
fi

echo "============================================"
echo " Doctor.Song - Stage 1: Pretraining (PT)"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: Full-Parameter"
echo " HF Endpoint: ${HF_ENDPOINT:-https://huggingface.co}"
echo " Start Time: $(date)"
echo "============================================"

time CUDA_VISIBLE_DEVICES=0 python3 training/pretraining.py \
    --model_name_or_path Qwen/Qwen3.5-0.8B-Base \
    --train_file_dir ./data/pretrain \
    --validation_file_dir ./data/pretrain \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_peft False \
    --seed 42 \
    --max_train_samples -1 \
    --max_eval_samples 50 \
    --num_train_epochs 1.0 \
    --learning_rate 5e-5 \
    --warmup_steps 10 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 5 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --block_size 512 \
    --packing True \
    --output_dir ./outputs/pt \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song PT - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Peak GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
echo "============================================"
echo "[Doctor.Song] Stage: PT | $(date) | Exit: $EXIT_CODE" >> training_timeline.log
