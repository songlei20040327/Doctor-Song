#!/bin/bash
# Doctor.Song - llama.cpp GGUF Server Launcher
# Model: Qwen3.5 + LoRA (Medical SFT + DPO) -> Q4_K_M GGUF (505MB)
# Performance: ~63 tok/s on Apple M2 with Metal GPU

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/outputs/final-gguf"
MODEL_FILE="$MODEL_DIR/doctor-song-Q4_K_M.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: GGUF model not found at $MODEL_FILE"
    echo "Run the conversion first: python3 convert_hf_to_gguf.py ..."
    exit 1
fi

echo "Starting Doctor.Song GGUF server..."
echo "Model: $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
echo "API: http://localhost:8000/v1/chat/completions"
echo ""

llama-server \
    -m "$MODEL_FILE" \
    --port 8000 \
    --host 127.0.0.1 \
    -ngl 99 \
    --ctx-size 4096 \
    --no-webui \
    --threads 4 \
    --alias doctor-song
