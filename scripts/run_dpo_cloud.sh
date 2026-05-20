#!/bin/bash
# ============================================================================
# Doctor.Song — 云端 DPO 训练全流程脚本
# 硬件要求：RTX 3090 / A5000 / A6000 (24GB+ VRAM)
#
# 上传到云端的内容（仅代码+数据，~5MB）：
#   tar czf dpo_cloud.tar.gz \
#       training/ training_utils.py training/template.py training/tool_utils.py \
#       tools/merge_peft_adapter.py \
#       scripts/run_dpo_cloud.sh \
#       data/reward_filtered/
#
# 模型从 HF 自动下载，无需上传 2.8GB 权重
# ============================================================================
set -euo pipefail

# ============================
# 配置
# ============================
FAST_DIR="/home/featurize/data"            # 高速本地盘
HF_CACHE="$FAST_DIR/hf_cache"
DATA_DIR="$FAST_DIR/dpo_data"               # 训练数据
OUTPUT_LORA="$FAST_DIR/dpo-lora"            # LoRA 输出
OUTPUT_MERGED="$FAST_DIR/dpo-merge"         # 合并模型
SYNC_DIR="/home/featurize/work"             # 同步盘（持久化）
SAVE_TO_SYNC=true

# HF 模型标识
BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
HF_REPO="Songlei327/Doctor.Song-0.8B-Medical"

echo "============================================"
echo " Doctor.Song DPO Training — Cloud Edition v2"
echo " 数据: 3848 条纯医学偏好对 (已筛选)"
echo " 配置: LoRA r=8, β=0.01, max_steps=300"
echo " Start: $(date)"
echo "============================================"

# ============================
# Step 1: 环境检查 & 依赖
# ============================
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    echo "GPU: $GPU_NAME ($GPU_MEM MiB)"
fi

mkdir -p "$FAST_DIR" "$HF_CACHE" "$DATA_DIR"
export HF_HOME="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_DATASETS_CACHE="$HF_CACHE"

# 绕过系统代理（HF 下载走直连或镜像，不走 SOCKS）
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY 2>/dev/null || true

echo ">>> 创建干净 Python 环境..."
if command -v conda &>/dev/null; then
    conda create -n doctor-song python=3.11 -y 2>/dev/null
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate doctor-song
elif command -v python3 -m venv &>/dev/null; then
    python3 -m venv "$FAST_DIR/venv"
    source "$FAST_DIR/venv/bin/activate"
fi
echo "  Python: $(which python3)"

echo ">>> 安装依赖..."
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
    pip install -q torch torchvision 2>/dev/null
pip install -q "transformers==4.51.0" "peft==0.14.0" "trl>=0.12"
pip install -q "huggingface-hub>=0.26,<1.0" accelerate loguru datasets
echo "✅ 依赖就绪"

# 自检
echo ">>> 环境自检..."
python3 -c "
import torch; print(f'  torch {torch.__version__}  ✅')
import transformers; print(f'  transformers {transformers.__version__}  ✅')
import peft; print(f'  peft {peft.__version__}  ✅')
import trl; print(f'  trl {trl.__version__}  ✅')
from training.utils import setup_tokenizer, find_all_linear_names; print('  training.utils  ✅')
from training.template import get_conv_template; print('  training.template  ✅')
" || { echo "❌ 自检失败"; exit 1; }

# HF 镜像（国内服务器）
if ! curl -s --connect-timeout 3 https://huggingface.co > /dev/null 2>&1; then
    export HF_ENDPOINT=https://hf-mirror.com
    echo "HF 镜像: hf-mirror.com"
fi

# ============================
# Step 2: 准备基座模型 (SFT-merge)
# ============================
echo ""
echo ">>> Step 2: 准备 SFT 基座模型..."

SFT_MERGE="$FAST_DIR/sft-merge"

if [ -f "$SFT_MERGE/model.safetensors" ]; then
    echo "✅ SFT 模型已存在: $SFT_MERGE"
else
    echo "从 HF 下载基座 + LoRA 适配器，合并..."

    # 下载 Qwen3.5-0.8B-Base（直接下载所有文件，不需要 git lfs）
    echo "  下载基座模型 Qwen3.5-0.8B-Base (约1.5GB)..."
    pip install -q huggingface_hub 2>/dev/null || true
    python3 -c "
import sys
from huggingface_hub import snapshot_download
print('  正在下载...')
try:
    snapshot_download('$BASE_MODEL', local_dir='$FAST_DIR/base-model', max_workers=4)
except Exception as e:
    print(f'  下载失败: {e}')
    sys.exit(1)
print('  基座模型下载完成')
"

    # 下载 SFT LoRA 适配器 (只有 40MB)
    echo "  下载 SFT LoRA 适配器 (40MB)..."
    python3 -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', allow_patterns='lora/sft-lora/*',
                  local_dir='$FAST_DIR')
print('  LoRA 下载完成')
"

    # 合并
    echo "  合并 LoRA → 完整模型..."
    python3 tools/merge_peft_adapter.py \
        --base_model "$FAST_DIR/base-model" \
        --lora_model "$FAST_DIR/lora/sft-lora" \
        --output_dir "$SFT_MERGE" \
        --tokenizer_path "$FAST_DIR/base-model"

    echo "✅ SFT 模型就绪: $SFT_MERGE"
    du -sh "$SFT_MERGE"
fi

# ============================
# Step 3: 准备 DPO 数据
# ============================
echo ""
echo ">>> Step 3: 准备 DPO 训练数据..."

# 数据已经随代码包上传到 data/reward_filtered/
LOCAL_DATA="./data/reward_filtered"
if [ -d "$LOCAL_DATA" ] && [ -f "$LOCAL_DATA/train.jsonl" ]; then
    cp "$LOCAL_DATA/train.jsonl" "$DATA_DIR/train.jsonl"
    echo "  使用上传的筛选数据: $(wc -l < $DATA_DIR/train.jsonl) 条"
else
    echo "❌ 未找到 data/reward_filtered/train.jsonl"
    echo "   请确保已将筛选后的 DPO 数据随代码包上传"
    exit 1
fi

# ============================
# Step 4: DPO 训练
# ============================
echo ""
echo ">>> Step 4: DPO LoRA 训练"
echo "    配置: β=0.01 | 纯医学 3848 条 | max_steps=300 | batch=2×4"
echo "    预估耗时: 40-60 分钟 (RTX 3090)"
echo ""

CUDA_VISIBLE_DEVICES=0 python3 training/dpo_training.py \
    --model_name_or_path "$SFT_MERGE" \
    --train_file_dir "$DATA_DIR" \
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
    --output_dir "$OUTPUT_LORA" \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to none \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --cache_dir "$HF_CACHE"

DPO_EXIT=$?
echo ""
echo "DPO 训练完成 (exit: $DPO_EXIT)"

if [ $DPO_EXIT -ne 0 ]; then
    echo "❌ DPO 训练失败，退出"
    exit $DPO_EXIT
fi

# ============================
# Step 5: 合并 LoRA
# ============================
echo ""
echo ">>> Step 5: 合并 LoRA 适配器..."

python3 tools/merge_peft_adapter.py \
    --base_model "$SFT_MERGE" \
    --lora_model "$OUTPUT_LORA" \
    --output_dir "$OUTPUT_MERGED" \
    --tokenizer_path "$SFT_MERGE"

echo "✅ 合并完成: $OUTPUT_MERGED"
du -sh "$OUTPUT_MERGED"

# ============================
# Step 6: 保存到同步盘
# ============================
if [ "$SAVE_TO_SYNC" = true ] && [ -d "$SYNC_DIR" ]; then
    echo ""
    echo ">>> Step 6: 备份 LoRA 到同步盘..."

    LORA_BACKUP="$SYNC_DIR/dpo-lora-v2"
    mkdir -p "$LORA_BACKUP"
    cp -r "$OUTPUT_LORA"/* "$LORA_BACKUP"/

    # 也保存训练损失曲线数据
    if [ -f "$OUTPUT_LORA/all_results.json" ]; then
        cp "$OUTPUT_LORA/all_results.json" "$SYNC_DIR/dpo_results_v2.json"
    fi

    echo "✅ LoRA 适配器已备份: $LORA_BACKUP"
    echo "   实例销毁后数据不丢失"
else
    echo "⚠️  未备份！请手动下载 $OUTPUT_LORA"
fi

# ============================
# 完成
# ============================
echo ""
echo "============================================"
echo " 🎉 Doctor.Song DPO v2 训练完成！"
echo "============================================"
echo " End: $(date)"
echo ""
echo "产出:"
echo "  LoRA:  $OUTPUT_LORA"
echo "  Merge: $OUTPUT_MERGED"
echo "  备份:  $SYNC_DIR/dpo-lora-v2"
echo ""
echo "评测: python3 run_full_eval.py"
echo "============================================"
