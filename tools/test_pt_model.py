# -*- coding: utf-8 -*-
"""本地测试 PT 模型 — 续写能力"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/pt")

# 自动选设备：CUDA > MPS > CPU
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32  # MPS 不支持 bf16
else:
    device, dtype = "cpu", torch.float32

print(f"Device: {device}, dtype: {dtype}")
print(f"Loading model: {MODEL_PATH} ...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

prompts = [
    "头痛的常见病因包括",
    "糖尿病患者的饮食应该注意",
    "抗生素的作用机制是",
    "高血压的诊断标准是",
    "心肺复苏的步骤包括",
]

print(f"\n{'='*60}")
print("PT 模型续写测试")
print("="*60)

for p in prompts:
    inputs = tokenizer(p, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n>>> {p}")
    print(result[len(p):].strip())
