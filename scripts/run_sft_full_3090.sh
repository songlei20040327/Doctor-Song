#!/bin/bash
# ============================================================================
# Doctor.Song 阶段2：监督微调 (SFT) - 全参数版本
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：全参数训练 (use_peft=False)
# 数据：医学思维链 SFT 数据 (含 <think>/<answer> 推理格式)
# 模板：qwen3_5 (原生思考模式)
# 产出：outputs-sft-doctor-song-full/
#
# 这是 LoRA vs Full-Parameter 对比实验的 Full 组
# ============================================================================

echo "============================================"
echo " Doctor.Song - Stage 2a: SFT (Full-Parameter)"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: Full-Parameter"
echo " Template: qwen3_5 (thinking mode enabled)"
echo " Start Time: $(date)"
echo "============================================"

time CUDA_VISIBLE_DEVICES=0 python3 training/supervised_finetuning.py \
    --model_name_or_path ./outputs/pt \
    --train_file_dir ./data/sft \
    --validation_file_dir ./data/sft \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft False \
    --max_train_samples -1 \
    --max_eval_samples 100 \
    --model_max_length 2048 \
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
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 4 \
    --output_dir ./outputs/sft-full \
    --ddp_timeout 30000 \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing True \
    --template_name qwen3_5 \
    --cache_dir ./cache

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song SFT-Full - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Peak GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
echo "============================================"
echo "[Doctor.Song] Stage: SFT-Full | $(date) | Exit: $EXIT_CODE" >> training_timeline.log
