#!/bin/bash
# ============================================================================
# Doctor.Song - Gradio Web Demo
# 展示医学推理可视化（折叠/展开 think 过程）
# 类似 DeepSeek-R1 的展示方式
# 用法：bash scripts/run_gradio_3090.sh
# 访问：http://localhost:8081
# ============================================================================

echo "============================================"
echo " Doctor.Song - Gradio Web Demo"
echo " Model: Doctor.Song-0.8B-Medical"
echo " URL: http://localhost:8081"
echo " Start Time: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python3 demo/gradio_demo.py \
    --base_model ./outputs/final \
    --template_name qwen3_5_nothink \
    --system_prompt "你是 Doctor.Song，一位专业的医学AI助手。请给出准确、安全的医学建议。如果问题超出你的知识范围或涉及紧急情况，请建议用户咨询专业医生。" \
    --port 8081

echo ""
echo "Doctor.Song Gradio Demo stopped."
echo "[Doctor.Song] Stage: Gradio Demo | $(date)" >> training_timeline.log
