#!/bin/bash
# ============================================================================
# Doctor.Song 阶段3：偏好对齐 (DPO) - 医学安全对齐
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：LoRA (r=8, alpha=16) — 显存安全，全参数DPO需要 ~23GB
# 数据：高质量医学+通用偏好对 (经过长度/质量过滤)
# 产出：outputs-dpo-doctor-song-lora/
# ============================================================================

# ---- 网络配置：国内优先用镜像 ----
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null 2>&1; then
    echo "HF 直连可用"
else
    echo "HF 不可用，切换到镜像"
    export HF_ENDPOINT=https://hf-mirror.com
fi

echo "============================================"
echo " Doctor.Song - Stage 4: DPO Safety Alignment"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: LoRA (r=8, alpha=16)"
echo " Input: ./outputs/sft-merge"
echo " Start Time: $(date)"
echo "============================================"

# 0.8B 模型在 3090 上显存占用 ~4GB，24GB 绰绰有余
# batch_size=2, grad_accum=4 → effective_batch=8，吞吐翻倍
time CUDA_VISIBLE_DEVICES=0 python3 training/dpo_training.py \
    --model_name_or_path  ./outputs/sft-merge\
    --train_file_dir ./data/reward_filtered \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_peft True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --max_train_samples -1 \
    --max_eval_samples 50 \
    --validation_split_percentage 5 \
    --max_steps 300 \
    --logging_steps 10 \
    --eval_steps 50 \
    --save_steps 100 \
    --max_source_length 1024 \
    --max_target_length 1024 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 20 \
    --beta 0.01 \
    --loss_type sigmoid \
    --gradient_accumulation_steps 4 \
    --template_name qwen3_5_nothink \
    --output_dir ./outputs/dpo-lora \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --cache_dir ./cache

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song DPO-LoRA - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Output: outputs-dpo-doctor-song-lora/"
echo "============================================"
echo "[Doctor.Song] Stage: DPO-LoRA | $(date) | Exit: $EXIT_CODE" >> training_timeline.log

# 提示 merge 命令
echo ""
echo "DPO used LoRA training. To use the model:"
echo "  1. Merge: python tools/merge_simple.py"
echo "     (set LORA='outputs-dpo-doctor-song-lora', OUT='outputs-dpo-doctor-song-merged')"
echo "  2. Or use with API server that supports LoRA loading."
echo "Final Model: Doctor.Song-0.8B-Medical"
