#!/bin/bash
# ============================================================================
# Doctor.Song 阶段3：GRPO 推理链强化
# 硬件：单卡 RTX 3090 (24GB)
# 训练方式：QLoRA (4-bit量化 + LoRA, 省显存以同时保留rollout模型)
# 奖励函数：
#   - format_reward (权重0.3): 检查 <think>...</think><answer>...</answer> 格式
#   - accuracy_reward (权重0.7): 医学MCQ正确性检查
# 思路：类 DeepSeek-R1 的 R1-Zero 方式，用RL强化推理链质量
# 产出：outputs-grpo-doctor-song/
# ============================================================================

echo "============================================"
echo " Doctor.Song - Stage 3: GRPO Reasoning RL"
echo " GPU: Single RTX 3090 (24GB)"
echo " Training Mode: QLoRA (4-bit + LoRA r=16)"
echo " Reward: format_reward + accuracy_reward"
echo " Start Time: $(date)"
echo "============================================"

time CUDA_VISIBLE_DEVICES=0 python3 training/grpo_training.py \
    --model_name_or_path outputs-sft-doctor-song-full \
    --train_file_dir data/grpo \
    --train_samples -1 \
    --max_steps -1 \
    --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 5 \
    --output_dir outputs-grpo-doctor-song \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --beta 0.001 \
    --learning_rate 5.0e-7 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --use_vllm False \
    --logging_steps 10 \
    --use_peft True \
    --qlora True \
    --load_in_4bit True \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --gradient_accumulation_steps 1 \
    --max_completion_length 1024

EXIT_CODE=$?
echo ""
echo "============================================"
echo " Doctor.Song GRPO - Exit Code: $EXIT_CODE"
echo " End Time: $(date)"
echo " Peak GPU Memory:"
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi not available"
echo "============================================"
echo "[Doctor.Song] Stage: GRPO | $(date) | Exit: $EXIT_CODE" >> training_timeline.log
