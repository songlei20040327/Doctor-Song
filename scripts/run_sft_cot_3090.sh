#!/bin/bash
# ============================================================================
# Doctor.Song 阶段5：CoT 推理链注入 (SFT-LoRA)
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：LoRA (r=16, alpha=32) — 推理链学习需要稍大的 rank
# 数据：DeepSeek 生成的 CoT 推理链数据 (3,000条)
# 模板：qwen3_5 (开启思考模式)
# 产出：outputs-sft-cot-doctor-song-lora/
# ============================================================================

# ---- 网络配置 ----
if curl -s --connect-timeout 5 https://huggingface.co > /dev/null 2>&1; then
    echo "HF 直连可用"
else
    echo "HF 不可用，切换到镜像"
    export HF_ENDPOINT=https://hf-mirror.com
fi

echo "============================================"
echo " Doctor.Song - Stage 5: CoT SFT (LoRA)"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: LoRA (r=16, alpha=32)"
echo " Template: qwen3_5 (thinking mode)"
echo " Data: 3,000 CoT samples"
echo " Start Time: $(date)"
echo "============================================"

time CUDA_VISIBLE_DEVICES=0 python3 training/supervised_finetuning.py \
    --model_name_or_path  ./outputs/dpo-merge\
    --train_file_dir ./data/sft_cot \
    --validation_file_dir ./data/sft_cot \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --max_train_samples -1 \
    --max_eval_samples 50 \
    --model_max_length 2048 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 100 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --template_name qwen3_5 \
    --output_dir outputs/sft-cot \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --cache_dir ./cache

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song CoT-SFT - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Output: outputs-sft-cot-doctor-song-lora/"
echo "============================================"
echo "[Doctor.Song] Stage: CoT-SFT | $(date) | Exit: $EXIT_CODE" >> training_timeline.log

echo ""
echo "CoT-SFT used LoRA training. To use the model:"
echo "  Merge with: outputs-dpo-doctor-song-merged + outputs-sft-cot-doctor-song-lora"
echo "  Then: Doctor.Song-0.8B-Medical (会思考 + 安全对齐)"
