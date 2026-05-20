# -*- coding: utf-8 -*-
"""
Doctor.Song 数据准备脚本：下载、采样、格式化、清理Demo数据

用法:
    python tools/prepare_data.py --all              # 下载所有数据
    python tools/prepare_data.py --stage pt         # 仅下载PT数据
    python tools/prepare_data.py --stage sft        # 仅下载SFT数据
    python tools/prepare_data.py --stage dpo        # 仅下载DPO数据
    python tools/prepare_data.py --clean            # 仅清理Demo数据

数据量适配: 针对 Qwen3.5-0.8B (800M参数) 精心设计采样比例
    - PT:  ~30K 医学文档 + 30K 维基百科条目 = 60K 文档 (~150MB)
    - SFT: ~15K 医学中文QA + 5K 华佗对话 = 20K 条
    - DPO: ~4K 医学偏好对 + 3K 知乎偏好对 = 7K 条

数据来源:
    - shibing624/medical (HuggingFace, raw JSON) - 240万条医疗数据
    - pleisto/wikipedia-cn-20230720-filtered - 中文维基百科
    - FreedomIntelligence/HuatuoGPT-sft-data-v1 - 华佗医疗对话
    - liyucheng/zhihu_rlhf_3k - 中文知乎偏好数据
"""
import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# 设置随机种子保证可复现
random.seed(42)

# HuggingFace 原始JSON文件下载基础URL
HF_RAW_BASE = "https://huggingface.co/datasets/shibing624/medical/resolve/main"

# 数据文件映射
MEDICAL_FILES = {
    "pt": {
        "train_encyclopedia": f"{HF_RAW_BASE}/pretrain/train_encyclopedia.json",
        "medical_book_zh": f"{HF_RAW_BASE}/pretrain/medical_book_zh.json",
        "valid_encyclopedia": f"{HF_RAW_BASE}/pretrain/valid_encyclopedia.json",
        "test_encyclopedia": f"{HF_RAW_BASE}/pretrain/test_encyclopedia.json",
    },
    "sft": {
        "train_zh_0": f"{HF_RAW_BASE}/finetune/train_zh_0.json",
        "valid_zh_0": f"{HF_RAW_BASE}/finetune/valid_zh_0.json",
        "test_zh_0": f"{HF_RAW_BASE}/finetune/test_zh_0.json",
    },
    "dpo": {
        "train": f"{HF_RAW_BASE}/reward/train.json",
        "valid": f"{HF_RAW_BASE}/reward/valid.json",
        "test": f"{HF_RAW_BASE}/reward/test.json",
    },
}

# 各阶段采样量（针对0.8B模型优化）
SAMPLE_SIZES = {
    "pt_encyclopedia": 30000,   # 从百科全书取3万条
    "pt_medical_book": -1,      # 医学书籍全部保留(约2万条)
    "pt_wikipedia": 30000,      # 维基百科取3万条
    "sft_medical_zh": 15000,    # 医学中文QA取1.5万条
    "sft_huatuo": 5000,         # 华佗对话取5千条
    "dpo_medical": -1,          # 医学偏好全部保留(约4千条)
    "dpo_zhihu": -1,            # 知乎偏好全部保留(约3千条)
}

# 下载文件
def download_file(url: str, dest: str) -> bool:
    """下载单个文件，带进度提示"""
    import urllib.request

    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  [skip] {os.path.basename(dest)} ({size_mb:.1f} MB) already exists")
        return True

    print(f"  [downloading] {os.path.basename(dest)} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"OK ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# 清楚demo数据
def clean_demo_data():
    """清理Demo示例数据"""
    print("\n" + "=" * 50)
    print("  Cleaning demo data...")
    print("=" * 50)

    # PT demo data (天龙八部、英文文章)
    pt_demo_files = [
        "data/pretrain/tianlongbabu.txt",
        "data/pretrain/fever.txt",
        "data/pretrain/en_article_tail500.txt",
    ]
    for f in pt_demo_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  [deleted] {f}")

    # SFT demo data
    sft_demo_files = [
        "data/sft/medical_sft_1K_format.jsonl",
        "data/sft/sharegpt_zh_1K_format.jsonl",
    ]
    for f in sft_demo_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"  [deleted] {f}")

    # DPO demo data
    dpo_demo = "data/reward/dpo_zh_500.jsonl"
    if os.path.exists(dpo_demo):
        os.remove(dpo_demo)
        print(f"  [deleted] {dpo_demo}")

    # GRPO demo data (数学题)
    grpo_demo = "data/grpo/sample.jsonl"
    if os.path.exists(grpo_demo):
        os.remove(grpo_demo)
        print(f"  [deleted] {grpo_demo}")

    print("  Demo data cleaned.\n")


def prepare_pt_data(raw_dir: str):
    """准备预训练数据: 医学百科全书 + 医学书籍 + 维基百科"""
    print("\n" + "=" * 50)
    print("  Preparing PT (Pretraining) Data...")
    print("=" * 50)

    output_dir = "data/pretrain"
    os.makedirs(output_dir, exist_ok=True)

    all_texts = []

    # 1. 医学百科全书 (train_encyclopedia.json)
    fpath = os.path.join(raw_dir, "pretrain", "train_encyclopedia.json")
    if os.path.exists(fpath):
        print(f"  Loading: train_encyclopedia.json ...")
        texts = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if "text" in obj and obj["text"].strip():
                        texts.append(obj["text"].strip())
                except json.JSONDecodeError:
                    continue
        if SAMPLE_SIZES["pt_encyclopedia"] > 0 and len(texts) > SAMPLE_SIZES["pt_encyclopedia"]:
            texts = random.sample(texts, SAMPLE_SIZES["pt_encyclopedia"])
        all_texts.extend(texts)
        print(f"    → {len(texts)} documents")

    # 2. 医学书籍 (medical_book_zh.json)
    fpath = os.path.join(raw_dir, "pretrain", "medical_book_zh.json")
    if os.path.exists(fpath):
        print(f"  Loading: medical_book_zh.json ...")
        texts = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if "text" in obj and obj["text"].strip():
                        texts.append(obj["text"].strip())
                except json.JSONDecodeError:
                    continue
        all_texts.extend(texts)
        print(f"    → {len(texts)} documents")

    # 3. 中文维基百科 (从datasets库加载，下载失败则跳过)
    print(f"  Loading: wikipedia-cn-20230720-filtered ...")
    try:
        from datasets import load_dataset
        wiki = load_dataset("pleisto/wikipedia-cn-20230720-filtered", split="train", streaming=True)
        wiki_texts = []
        for i, item in enumerate(wiki):
            if "text" in item and item["text"].strip():
                wiki_texts.append(item["text"].strip())
            if SAMPLE_SIZES["pt_wikipedia"] > 0 and len(wiki_texts) >= SAMPLE_SIZES["pt_wikipedia"]:
                break
        all_texts.extend(wiki_texts)
        print(f"    → {len(wiki_texts)} documents")
    except Exception as e:
        print(f"    WARNING: Wikipedia download failed: {e}")
        print(f"    Skipping Wikipedia data (non-critical)")

    # 保存为 .txt 格式（每行一篇文档）
    if all_texts:
        # 打乱
        random.shuffle(all_texts)
        output_file = os.path.join(output_dir, "medical_pretrain.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for text in all_texts:
                # 移除换行符，每行一篇文档
                clean_text = text.replace("\n", " ").replace("\r", " ").strip()
                if clean_text:
                    f.write(clean_text + "\n")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n  ✓ Saved: {output_file} ({len(all_texts):,} docs, {size_mb:.1f} MB)")
    else:
        print(f"\n  WARNING: No PT data prepared!")
        print(f"  You may need to set HF_TOKEN or check network connection.")

    return len(all_texts)


def convert_finetune_to_sharegpt(row: dict) -> dict:
    """将 shibing624/medical finetune 格式转为 ShareGPT 格式"""
    instruction = row.get("instruction", "").strip()
    input_text = row.get("input", "").strip()
    output_text = row.get("output", "").strip()

    # 合并 instruction 和 input 作为用户问题
    if input_text:
        user_query = f"{instruction}\n{input_text}"
    else:
        user_query = instruction

    return {
        "conversations": [
            {"from": "human", "value": user_query},
            {"from": "gpt", "value": output_text},
        ]
    }


def prepare_sft_data(raw_dir: str):
    """准备SFT数据: 医学中文QA + 华佗医疗对话"""
    print("\n" + "=" * 50)
    print("  Preparing SFT (Supervised Fine-Tuning) Data...")
    print("=" * 50)

    output_dir = "data/sft"
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []

    # 1. 医学中文QA (train_zh_0.json)
    fpath = os.path.join(raw_dir, "finetune", "train_zh_0.json")
    if os.path.exists(fpath):
        print(f"  Loading: train_zh_0.json ...")
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("instruction", "").strip() and obj.get("output", "").strip():
                        sharegpt_obj = convert_finetune_to_sharegpt(obj)
                        samples.append(sharegpt_obj)
                except json.JSONDecodeError:
                    continue
        if SAMPLE_SIZES["sft_medical_zh"] > 0 and len(samples) > SAMPLE_SIZES["sft_medical_zh"]:
            samples = random.sample(samples, SAMPLE_SIZES["sft_medical_zh"])
        all_samples.extend(samples)
        print(f"    → {len(samples)} samples")

    # 2. 华佗医疗对话
    print(f"  Loading: HuatuoGPT-sft-data-v1 ...")
    try:
        from datasets import load_dataset
        huatuo = load_dataset("FreedomIntelligence/HuatuoGPT-sft-data-v1", split="train", streaming=True)
        huatuo_samples = []
        for item in huatuo:
            # HuatuoGPT format varies, try to extract QA pairs
            conversations = []
            if "messages" in item:
                # Direct messages format
                for msg in item["messages"]:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        conversations.append({"from": "human", "value": content})
                    elif role == "assistant":
                        conversations.append({"from": "gpt", "value": content})
            elif "instruction" in item and "output" in item:
                conversations = [
                    {"from": "human", "value": item.get("instruction", "")},
                    {"from": "gpt", "value": item.get("output", "")},
                ]
            if conversations:
                huatuo_samples.append({"conversations": conversations})
            if SAMPLE_SIZES["sft_huatuo"] > 0 and len(huatuo_samples) >= SAMPLE_SIZES["sft_huatuo"]:
                break
        all_samples.extend(huatuo_samples)
        print(f"    → {len(huatuo_samples)} samples")
    except Exception as e:
        print(f"    WARNING: HuatuoGPT download failed: {e}")
        print(f"    Skipping HuatuoGPT data (non-critical)")

    # 保存
    if all_samples:
        random.shuffle(all_samples)
        output_file = os.path.join(output_dir, "medical_sft_cot.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n  ✓ Saved: {output_file} ({len(all_samples):,} samples, {size_mb:.1f} MB)")
        print(f"  NOTE: This data does NOT yet contain <think>/<answer> CoT format.")
        print(f"  To add reasoning chains, run: python tools/generate_cot.py (coming soon)")
    else:
        print(f"\n  WARNING: No SFT data prepared!")

    return len(all_samples)


def convert_reward_to_dpo(row: dict) -> dict:
    """将 shibing624/medical reward 格式转为 DPO 格式"""
    question = row.get("question", "").strip()
    chosen = row.get("response_chosen", "").strip()
    rejected = row.get("response_rejected", "").strip()
    return {
        "conversations": [{"from": "human", "value": question}],
        "chosen": chosen,
        "rejected": rejected,
    }


def filter_dpo_sample(sample: dict) -> bool:
    """DPO数据质量过滤：确保chosen/rejected长度合理，对比有意义"""
    chosen = sample.get("chosen", "")
    rejected = sample.get("rejected", "")
    question = sample.get("conversations", [{}])[0].get("value", "")

    MIN_CHOSEN = 20      # chosen至少20字（医学回答可能简短但专业）
    MIN_REJECTED = 10     # rejected至少10字
    MIN_QUESTION = 6      # 问题至少6字
    MAX_LEN_RATIO = 5.0   # chosen/rejected长度比不超过5倍

    if len(question) < MIN_QUESTION:
        return False
    if len(chosen) < MIN_CHOSEN:
        return False
    if len(rejected) < MIN_REJECTED:
        return False
    # 避免极端长度偏差
    if len(rejected) > 0:
        ratio = len(chosen) / len(rejected)
        if ratio > MAX_LEN_RATIO or ratio < (1.0 / MAX_LEN_RATIO):
            return False
    return True


def prepare_dpo_data(raw_dir: str):
    """准备DPO数据: 医学偏好对 + 知乎RLHF偏好对 + DPO-En-Zh-20k"""
    print("\n" + "=" * 50)
    print("  Preparing DPO (Preference) Data...")
    print("  Quality filters: chosen>={}chars, rejected>={}chars, ratio<={}x"
          .format(30, 10, 5.0))
    print("=" * 50)

    output_dir = "data/reward"
    os.makedirs(output_dir, exist_ok=True)

    all_samples = []
    stats = {"medical_raw": 0, "zhihu_raw": 0, "dpo20k_raw": 0}

    # 1. 医学偏好对 (shibing624/medical reward)
    fpath = os.path.join(raw_dir, "reward", "train.json")
    if os.path.exists(fpath):
        print(f"  Loading: shibing624/medical reward/train.json ...")
        samples = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    if obj.get("question") and obj.get("response_chosen") and obj.get("response_rejected"):
                        dpo_obj = convert_reward_to_dpo(obj)
                        samples.append(dpo_obj)
                except json.JSONDecodeError:
                    continue
        stats["medical_raw"] = len(samples)
        print(f"    → {len(samples)} raw pairs")
        all_samples.extend(samples)

    # 2. 知乎RLHF偏好对 (安全性对齐)
    print(f"  Loading: liyucheng/zhihu_rlhf_3k ...")
    try:
        from datasets import load_dataset
        zhihu = load_dataset("liyucheng/zhihu_rlhf_3k", split="train", streaming=True)
        zhihu_samples = []
        for item in zhihu:
            question = item.get("question", "") or item.get("prompt", "")
            chosen = item.get("chosen", "") or item.get("response_chosen", "")
            rejected = item.get("rejected", "") or item.get("response_rejected", "")
            if question and chosen and rejected:
                zhihu_samples.append({
                    "conversations": [{"from": "human", "value": question}],
                    "chosen": chosen,
                    "rejected": rejected,
                })
        stats["zhihu_raw"] = len(zhihu_samples)
        print(f"    → {len(zhihu_samples)} raw pairs")
        all_samples.extend(zhihu_samples)
    except Exception as e:
        print(f"    WARNING: zhihu_rlhf_3k download failed: {e}")

    # 3. DPO-En-Zh-20k 中英文偏好数据（中文子集，补充安全性/无害性偏好）
    print(f"  Loading: shibing624/DPO-En-Zh-20k-Preference (zh) ...")
    try:
        from datasets import load_dataset
        dpo20k = load_dataset("shibing624/DPO-En-Zh-20k-Preference", "zh", split="train", streaming=True)
        dpo20k_samples = []
        for item in dpo20k:
            prompt = item.get("prompt", "") or item.get("question", "")
            chosen = item.get("chosen", "") or item.get("response_chosen", "")
            rejected = item.get("rejected", "") or item.get("response_rejected", "")
            if prompt and chosen and rejected:
                dpo20k_samples.append({
                    "conversations": [{"from": "human", "value": prompt}],
                    "chosen": chosen,
                    "rejected": rejected,
                })
            if len(dpo20k_samples) >= 4000:  # 中文子集采样4000条
                break
        stats["dpo20k_raw"] = len(dpo20k_samples)
        print(f"    → {len(dpo20k_samples)} raw pairs (sampled)")
        all_samples.extend(dpo20k_samples)
    except Exception as e:
        print(f"    WARNING: DPO-En-Zh-20k download failed: {e}")

    # 质量过滤
    raw_total = len(all_samples)
    all_samples = [s for s in all_samples if filter_dpo_sample(s)]
    filtered = raw_total - len(all_samples)

    # 保存
    if all_samples:
        random.shuffle(all_samples)
        output_file = os.path.join(output_dir, "medical_dpo.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"\n  ✓ Saved: {output_file} ({len(all_samples):,} pairs, {size_mb:.1f} MB)")
        print(f"  Filtered: {filtered}/{raw_total} removed ({100*filtered/max(raw_total,1):.0f}%)")
        print(f"  Source: medical={stats['medical_raw']} + zhihu={stats['zhihu_raw']} + dpo20k={stats['dpo20k_raw']}")
    else:
        print(f"\n  WARNING: No DPO data after filtering! ({raw_total} raw, {filtered} removed)")

    return len(all_samples)


def prepare_grpo_data():
    """准备GRPO数据: 从SFT数据构造医学问答（GRPO需要question+answer格式）"""
    print("\n" + "=" * 50)
    print("  Preparing GRPO Data...")
    print("=" * 50)

    output_dir = "data/grpo"
    os.makedirs(output_dir, exist_ok=True)

    # 从SFT数据中抽取简单的医学问答用于GRPO
    sft_file = "data/sft/medical_sft_cot.jsonl"
    if not os.path.exists(sft_file):
        print(f"  WARNING: SFT data not found at {sft_file}, skipping GRPO data.")
        return 0

    grpo_samples = []
    with open(sft_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                convs = obj.get("conversations", [])
                if len(convs) >= 2:
                    question = convs[0].get("value", "")
                    answer = convs[-1].get("value", "")
                    if question and answer and len(question) < 500:
                        grpo_samples.append({"question": question, "answer": answer})
            except json.JSONDecodeError:
                continue

    # 采样500条用于GRPO
    if len(grpo_samples) > 500:
        grpo_samples = random.sample(grpo_samples, 500)

    if grpo_samples:
        output_file = os.path.join(output_dir, "medical_grpo.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in grpo_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"  ✓ Saved: {output_file} ({len(grpo_samples)} samples)")
    else:
        print(f"  WARNING: No GRPO data prepared!")

    return len(grpo_samples)


def main():
    parser = argparse.ArgumentParser(description="Doctor.Song Data Preparation")
    parser.add_argument("--all", action="store_true", help="Download and prepare all data")
    parser.add_argument("--stage", type=str, choices=["pt", "sft", "dpo", "grpo"],
                        help="Prepare specific stage data")
    parser.add_argument("--clean", action="store_true", help="Clean demo data only")
    parser.add_argument("--raw_dir", type=str, default="./data/raw",
                        help="Directory for raw downloaded files")
    args = parser.parse_args()

    if not any([args.all, args.stage, args.clean]):
        parser.print_help()
        print("\nExamples:")
        print("  python tools/prepare_data.py --all          # Full preparation")
        print("  python tools/prepare_data.py --stage pt      # PT data only")
        print("  python tools/prepare_data.py --clean         # Clean demo data")
        return

    # 清理Demo数据
    clean_demo_data()
    if args.clean:
        print("Demo data cleaned. Use --all or --stage to download new data.")
        return

    # 确定要准备的阶段
    stages = ["pt", "sft", "dpo", "grpo"] if args.all else [args.stage]

    # 创建原始数据目录
    raw_dir = args.raw_dir
    os.makedirs(raw_dir, exist_ok=True)

    # 下载 shibing624/medical 原始JSON文件
    print("\n" + "=" * 50)
    print("  Downloading shibing624/medical raw files...")
    print("=" * 50)
    print("  (Raw JSON download due to datasets v4 incompatibility)")
    print(f"  Files will be saved to: {raw_dir}/\n")

    for stage_key in ["pt", "sft", "dpo"]:
        if stage_key in stages:
            for name, url in MEDICAL_FILES[stage_key].items():
                dest_dir = os.path.join(raw_dir, stage_key)
                os.makedirs(dest_dir, exist_ok=True)
                dest = os.path.join(dest_dir, f"{name}.json")
                download_file(url, dest)

    # 准备各阶段数据
    total = {"pt": 0, "sft": 0, "dpo": 0, "grpo": 0}

    if "pt" in stages:
        total["pt"] = prepare_pt_data(raw_dir)

    if "sft" in stages:
        total["sft"] = prepare_sft_data(raw_dir)

    if "dpo" in stages:
        total["dpo"] = prepare_dpo_data(raw_dir)

    if "grpo" in stages or args.all:
        total["grpo"] = prepare_grpo_data()

    # 总结
    print("\n" + "=" * 60)
    print("  Doctor.Song Data Preparation Complete!")
    print("=" * 60)
    print(f"  PT data:   {total['pt']:>8,} documents  → data/pretrain/medical_pretrain.txt")
    print(f"  SFT data:  {total['sft']:>8,} samples   → data/sft/medical_sft_cot.jsonl")
    print(f"  DPO data:  {total['dpo']:>8,} pairs     → data/reward/medical_dpo.jsonl")
    print(f"  GRPO data: {total['grpo']:>8,} samples   → data/grpo/medical_grpo.jsonl")
    print(f"\n  Raw downloads cached at: {raw_dir}/")
    print("\n  Next steps:")
    print("    1. (Optional) Generate CoT reasoning chains for SFT data")
    print("    2. Start training: bash scripts/run_pt_3090.sh")
    print("    3. Then: bash scripts/run_sft_full_3090.sh")
    print("    4. Then: bash scripts/run_sft_lora_3090.sh (comparison)")
    print("=" * 60)


if __name__ == "__main__":
    main()
