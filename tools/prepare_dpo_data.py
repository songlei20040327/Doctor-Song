#!/usr/bin/env python3
"""
从 shibing624/medical 下载官方 reward 数据并转换为 DPO 训练格式

官方格式: {question, response_chosen, response_rejected}
训练格式: {conversations: [{from: system, value: ...}, {from: human, value: ...}], chosen, rejected}

用法:
  python tools/prepare_dpo_data.py --output data/reward_medical --samples -1
"""

import json, os, argparse, urllib.request

SYSTEM_PROMPT = (
    "你是 Doctor.Song，一位专业的医学AI助手。"
    "请给出准确、安全、循证的医学建议。"
    "如果问题超出知识范围或涉及紧急情况，请建议就医。"
    "⚠ 你的建议仅供参考，不能替代专业医生诊断。"
)

# 官方 reward 数据集 URL（直接从 HuggingFace 下载原始 JSON）
HF_BASE = "https://huggingface.co/datasets/shibing624/medical/resolve/main/reward"
FILES = {
    "train": f"{HF_BASE}/train.json",
    "valid": f"{HF_BASE}/valid.json",
    "test": f"{HF_BASE}/test.json",
}
HF_MIRROR_BASE = "https://hf-mirror.com/datasets/shibing624/medical/resolve/main/reward"


def download_json(url, dest_path):
    """Download JSON file from URL to local path."""
    print(f"  Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"  Retrying with urllib... ({e})")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(dest_path, "wb") as f:
                f.write(resp.read())
    print(f"  Saved to {dest_path}")


def load_json_lines(filepath):
    """Load a JSON file that might be a list or JSONL."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if text.startswith("["):
        return json.loads(text)
    # JSONL format
    items = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def convert_to_sharegpt(question, response_chosen, response_rejected):
    """Convert official format to ShareGPT DPO format."""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": question},
        ],
        "chosen": response_chosen,
        "rejected": response_rejected,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/reward_medical", help="Output directory")
    parser.add_argument("--samples", type=int, default=-1, help="Number of samples (-1 = all)")
    parser.add_argument("--mirror", action="store_true", help="Use hf-mirror.com")
    args = parser.parse_args()

    base = HF_MIRROR_BASE if args.mirror else HF_BASE
    os.makedirs(args.output, exist_ok=True)

    all_items = []
    for split in ["train", "valid", "test"]:
        url = f"{base}/{split}.json"
        local = os.path.join(args.output, f"_{split}_raw.json")
        if not os.path.exists(local):
            download_json(url, local)
        items = load_json_lines(local)
        all_items.extend(items)
        print(f"  {split}: {len(items)} pairs")

    print(f"Total downloaded: {len(all_items)} pairs")

    # Convert and filter
    train_data = []
    valid_data = []

    for i, item in enumerate(all_items):
        question = item.get("question", "")
        chosen = item.get("response_chosen", "")
        rejected = item.get("response_rejected", "")

        if not question or not chosen or not rejected:
            continue
        if len(chosen) < 20 or len(rejected) < 20:
            continue

        converted = convert_to_sharegpt(question, chosen, rejected)

        if i % 10 == 0:
            valid_data.append(converted)
        else:
            train_data.append(converted)

    if args.samples > 0:
        train_data = train_data[:args.samples]

    train_file = os.path.join(args.output, "train.jsonl")
    valid_file = os.path.join(args.output, "valid.jsonl")

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Train: {len(train_data)} pairs -> {train_file}")
    print(f"✅ Valid: {len(valid_data)} pairs -> {valid_file}")
    print(f"\nData format check (first sample):")
    print(f"  Question: {train_data[0]['conversations'][1]['value'][:80]}...")
    print(f"  Chosen:   {train_data[0]['chosen'][:80]}...")
    print(f"  Rejected: {train_data[0]['rejected'][:80]}...")


if __name__ == "__main__":
    main()
