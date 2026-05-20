#!/usr/bin/env python3
"""
Doctor.Song 模型上传到 HuggingFace
底层本质：每 push 一个文件夹，HF API 在后台做 git add + git commit + git push

使用：python scripts/upload_to_hf.py
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

PROJECT = Path(__file__).resolve().parent.parent
REPO_NAME = "Doctor.Song-0.8B-Medical"          # 改成你的 HF 用户名/组织名
# 如果你有 HF 用户名叫 leong，完整 ID 就是 "leong/Doctor.Song-0.8B-Medical"

api = HfApi()

# ---- 创建仓库（如果不存在）----
print(f">>> 创建仓库: {REPO_NAME}")
try:
    create_repo(REPO_NAME, exist_ok=True)
    print("  ✅ 仓库就绪")
except Exception as e:
    print(f"  ⚠️  {e}")

# ---- 上传 FP32 完整模型 ----
print(f"\n>>> 上传 FP32 模型...")
upload_folder(
    folder_path=str(PROJECT / "outputs" / "final"),
    repo_id=REPO_NAME,
    # 这些大文件类型自动走 LFS
    allow_patterns=[
        "*.safetensors", "*.bin", "*.json", "*.txt",
        "tokenizer*", "vocab*", "*.model", "*.py",
        "config*", "generation_config.json",
    ],
    commit_message="Upload Doctor.Song-0.8B-Medical (FP32) — 2.82GB",
)

# ---- 上传模型卡片 ----
print(f"\n>>> 上传 README (模型卡)...")
api.upload_file(
    path_or_fileobj=str(PROJECT / "scripts" / "hf_model_card.md"),
    path_in_repo="README.md",
    repo_id=REPO_NAME,
    commit_message="Add model card",
)

print(f"""
============================================
 🎉 上传完成！
 模型地址: https://huggingface.co/{REPO_NAME}
============================================
""")
