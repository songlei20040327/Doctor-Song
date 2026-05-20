# -*- coding: utf-8 -*-
"""
从高质量 SFT 数据生成医学 CoT 推理链数据

用法:
  python tools/generate_cot.py --api_key YOUR_DEEPSEEK_KEY --sample 3000
  python tools/generate_cot.py --api_key YOUR_DEEPSEEK_KEY --sample 3000 --resume

数据流:
  data/sft/medical_sft.jsonl (7981条 QA)
      ↓ 过滤：answer>=100字 + question>=10字
      ↓ 采样 3000 条
      ↓ 并发调用 DeepSeek API 生成 <think> 推理链
      ↓
  data/sft/medical_sft_cot.jsonl (CoT 格式)
"""

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# DeepSeek API 配置
DEEPSEEK_BASE = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"  # 便宜且强

SYSTEM_PROMPT = """你是一位资深医学专家。请为以下医学问答生成诊断推理过程。

要求：
1. 输出格式必须是:
<think>
[你的推理过程]
</think>
<answer>
[原始回答，保持完全不变]
</answer>

2. 推理过程应该包含：
   - 患者主诉总结
   - 需要鉴别的疾病（列出2-3种可能性）
   - 关键症状分析和排除逻辑
   - 最终诊断思路和建议依据

3. <answer> 部分必须原封不动复制下面的"原始回答"，一字不改。

4. <think> 部分控制在 150-400 字，简洁专业。"""


def filter_high_quality(samples, min_q=10, min_a=100):
    """筛选高质量样本：问题有实质内容，回答足够长"""
    result = []
    for s in samples:
        convs = s.get("conversations", [])
        if len(convs) < 2:
            continue
        q = convs[0].get("value", "").strip()
        a = convs[-1].get("value", "").strip()
        if len(q) >= min_q and len(a) >= min_a:
            result.append({"question": q, "answer": a})
    return result


def build_cot_prompt(question, answer):
    return f"""## 患者问题
{question}

## 原始回答
{answer}

请为以上医学问答生成推理过程。记住：<answer> 部分必须保持原始回答完全不变。"""


def call_deepseek(client, model, question, answer, max_retries=3):
    """调用 DeepSeek 生成推理链"""
    user_msg = build_cot_prompt(question, answer)
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            content = resp.choices[0].message.content
            # 验证格式
            if "<think>" in content and "<answer>" in content:
                return content
            # 重试
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None
    return None


def extract_answer_from_cot(cot_text):
    """提取 <answer> 部分"""
    if "<answer>" in cot_text and "</answer>" in cot_text:
        start = cot_text.index("<answer>") + len("<answer>")
        end = cot_text.index("</answer>")
        return cot_text[start:end].strip()
    return cot_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True, help="DeepSeek API key")
    parser.add_argument("--model", default="deepseek-chat", help="模型名 (默认 deepseek-chat)")
    parser.add_argument("--sample", type=int, default=3000, help="采样数量")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    parser.add_argument("--resume", action="store_true", help="断点续传")
    parser.add_argument("--input", default="data/sft/medical_sft.jsonl")
    parser.add_argument("--output", default="data/sft/medical_sft_cot.jsonl")
    args = parser.parse_args()

    random.seed(42)

    # 加载 SFT 数据
    print(f"Loading SFT data: {args.input}")
    samples = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    print(f"  Total: {len(samples)} samples")

    # 筛选高质量
    high_q = filter_high_quality(samples)
    print(f"  High quality (a>=100c): {len(high_q)} samples")

    # 采样
    if len(high_q) > args.sample:
        high_q = random.sample(high_q, args.sample)
    print(f"  Sampled: {len(high_q)} samples")

    # 断点续传
    done_questions = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                    done_questions.add(obj["conversations"][0]["value"])
                except:
                    pass
        print(f"  Resuming: {len(done_questions)} already done")

    pending = [(q["question"], q["answer"]) for q in high_q if q["question"] not in done_questions]
    print(f"  Pending: {len(pending)} samples")

    if not pending:
        print("All done!")
        return

    # 初始化 DeepSeek 客户端
    client = OpenAI(api_key=args.api_key, base_url=DEEPSEEK_BASE)

    # 并发生成
    success = 0
    fail = 0
    output_file = args.output
    mode = "a" if args.resume else "w"

    def process_one(item):
        q, a = item
        cot = call_deepseek(client, args.model, q, a)
        return q, a, cot

    print(f"\nGenerating CoT (workers={args.workers})...")
    with open(output_file, mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, item): item for item in pending}
            for i, future in enumerate(as_completed(futures)):
                q, a, cot = future.result()
                if cot:
                    # 保存为 ShareGPT 格式（含 <think> 推理）
                    sample = {
                        "conversations": [
                            {"from": "human", "value": q},
                            {"from": "gpt", "value": cot},
                        ]
                    }
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    fout.flush()
                    success += 1
                else:
                    fail += 1
                if (success + fail) % 50 == 0:
                    print(f"  [{success+fail}/{len(pending)}] success={success} fail={fail}")

    print(f"\n{'='*50}")
    print(f"  Done! success={success} fail={fail}")
    print(f"  Output: {output_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
