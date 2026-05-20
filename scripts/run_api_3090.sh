#!/bin/bash
# ============================================================================
# Doctor.Song - OpenAI 兼容 API 服务
# 提供 /v1/chat/completions 接口，方便接入任何前端
# 用法：bash scripts/run_api_3090.sh
# 访问：http://localhost:8000/docs (Swagger文档)
# 测试：curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{...}'
# ============================================================================

echo "============================================"
echo " Doctor.Song - OpenAI Compatible API"
echo " Model: Doctor.Song-0.8B-Medical"
echo " API Docs: http://localhost:8000/docs"
echo " Start Time: $(date)"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python3 demo/openai_api.py \
    --base_model ./outputs/final \
    --template_name qwen3_5_nothink \
    --server_port 8000 \
    --server_name 0.0.0.0

echo ""
echo "Doctor.Song API server stopped."
echo "[Doctor.Song] Stage: API Server | $(date)" >> training_timeline.log
