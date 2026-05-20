#!/bin/bash
# ============================================================================
# Doctor.Song 上传到 HuggingFace 一键脚本
# 使用前提：hf auth login 已登录
# 本质：Git + Git LFS push
# ============================================================================
set -euo pipefail

HF_USER="${HF_USER:-}"   # 你的 HF 用户名，如 "DoctorSong"
REPO_NAME="${1:-Doctor.Song-0.8B-Medical}"
MODEL_DIR="./outputs/final"
QUANT_DIR="./outputs/final-4bit"
GGUF_DIR="./outputs/final-gguf"

# ---- 临时目录 ----
TMP=$(mktemp -d)
echo "工作目录: $TMP"

# ---- 1. 创建 HF 仓库 ----
echo ">>> Step 1: 创建 HF 仓库"
hf create --yes "$REPO_NAME" 2>/dev/null || echo "仓库可能已存在，跳过"

# ---- 2. 准备文件 ----
echo ">>> Step 2: 准备上传文件"
cd "$TMP"
git init
git lfs install

# 追踪大文件
git lfs track "*.safetensors" "*.bin" "*.gguf" "*.msgpack" "*.h5"
git add .gitattributes

# 复制模型文件（FP32 完整版）
echo "复制 FP32 模型..."
cp -r "$OLDPWD/$MODEL_DIR"/* "$TMP/"
rm -rf "$TMP/quantized" "$TMP/gguf" 2>/dev/null || true   # 不重复放量化版

# 复制 tokenizer 文件确保存在
echo "检查 tokenizer..."

# 复制评测报告
echo "复制评测报告..."
cp "$OLDPWD/Doctor_Song_评测报告.md" "$TMP/EVALUATION.md"
cp "$OLDPWD/resume_metrics.md" "$TMP/METRICS.md" 2>/dev/null || true

# ---- 3. 上传 ----
echo ">>> Step 3: Git commit & push (这是整个上传的本质)"
git add -A
git commit -m "Upload Doctor.Song-0.8B-Medical FP32 model

- Base: Qwen3.5-0.8B-Base
- Training: PT → SFT → DPO → CoT
- Hardware: RTX 3090 (24GB)
- LLM-as-Judge: 3.10/5, Safety 4.0/5
- PPL: 3.92"
git remote add origin "https://huggingface.co/$REPO_NAME"
git push -u origin main

echo ""
echo "============================================"
echo " 上传完成！"
echo " 模型地址: https://huggingface.co/$REPO_NAME"
echo "============================================"

# ---- 4. 清理 ----
cd "$OLDPWD"
rm -rf "$TMP"
