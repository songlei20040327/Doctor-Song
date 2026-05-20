#!/bin/bash
# ============================================================================
# Doctor.Song 阶段2b：监督微调 (SFT) - LoRA 版本（对比实验）
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：LoRA (use_peft=True, r=8, alpha=16)
# 数据：医学思维链 SFT 数据 (含 <think>/<answer> 推理格式)
# 模板：qwen3_5_nothink (数据为纯问答，不含推理链)
# 产出：outputs-sft-doctor-song-lora/
#
# 后续如需加入推理链，用 CoT 数据 + qwen3_5 模板重跑
# ============================================================================

echo "============================================"
echo " Doctor.Song - Stage 2: SFT (LoRA)"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: LoRA (r=8, alpha=16)"
echo " Template: qwen3_5_nothink"
echo " Start Time: $(date)"
echo "============================================"

time CUDA_VISIBLE_DEVICES=0 python3 training/supervised_finetuning.py \
    --model_name_or_path ./outputs/pt \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 100 \
    --model_max_length 1024 \
    --num_train_epochs 3 \
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
    --save_total_limit 5 \
    --gradient_accumulation_steps 3 \
    --preprocessing_num_workers 4 \
    --dataloader_num_workers 4 \
    --output_dir ./outputs/sft-lora \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --template_name qwen3_5_nothink \
    --cache_dir ./cache

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song SFT-LoRA - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Peak GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
echo "============================================"
echo "[Doctor.Song] Stage: SFT-LoRA | $(date) | Exit: $EXIT_CODE" >> training_timeline.log
