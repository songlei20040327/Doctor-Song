#!/bin/bash
# 本地 Mac 启动 OpenAI 兼容 API (MPS 加速)
cd "$(dirname "$0")/.."

echo "Starting Doctor.Song API..."
echo "  API:  http://localhost:8000"
echo "  Docs: http://localhost:8000/docs"

python3 demo/openai_api.py \
    --base_model ./outputs/final \
    --template_name qwen3_5_nothink \
    --server_port 8000 \
    --server_name 0.0.0.0
