# -*- coding: utf-8 -*-
"""PT vs SFT 模型对比"""
import os, sys, torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = []
for arg in sys.argv[1:]:
    if ':' in arg:
        name, path = arg.split(':', 1)
        MODELS.append((name, path))
if not MODELS:
    MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/final")
    MODELS = [("Model", MODEL_PATH)]

QUESTIONS = [
    "我最近三天持续头痛，位置在太阳穴附近，伴有轻微恶心。没有发烧。可能是什么原因？需要注意什么？",
    "布洛芬和阿莫西林可以一起吃吗？我有点牙疼，同时医生说我有点细菌感染。",
    "体检报告显示谷丙转氨酶(ALT) 85 U/L（参考值0-40），可能是什么原因？",
    "我突然感觉胸口闷痛，持续了大约10分钟，伴有出冷汗。现在好一点了，需要去急诊吗？",
]

if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32
else:
    device, dtype = "cpu", torch.float32

results = {}

for name, path in MODELS:
    print(f"\n{'='*60}")
    print(f"  Loading {name} model... ({device})")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    results[name] = []

    for i, q in enumerate(QUESTIONS):
        t0 = time.time()

        if name in ("SFT", "DPO"):
            messages = [
                {"role": "system", "content": "你是 Doctor.Song，一位专业的医学AI助手。请给出准确、安全的医学建议。"},
                {"role": "user", "content": q},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"患者问题：{q}\n医生回答："

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        elapsed = time.time() - t0
        results[name].append(response.strip())
        print(f"\n  --- Q{i+1} ({elapsed:.0f}s): {q[:50]}... ---")
        print(f"  [{name}] {response.strip()[:500]}")
        print()

    del model
    if device == "mps":
        torch.mps.empty_cache()

model_names = [m[0] for m in MODELS]
print(f"\n{'='*60}")
print(f"  Summary: {' vs '.join(model_names)} 对比")
print(f"{'='*60}")

for i, q in enumerate(QUESTIONS):
    print(f"\n{'─'*60}")
    print(f"Q{i+1}: {q[:60]}...")
    print(f"{'─'*60}")
    for name in model_names:
        print(f"\n[{name} 回答]:")
        print(results[name][i][:400])
