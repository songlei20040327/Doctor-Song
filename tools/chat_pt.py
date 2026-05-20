# -*- coding: utf-8 -*-
"""PT 模型命令行对话（注意：PT只做续写，不是真对话）"""
import os
import torch
import readline  # 支持上下键历史
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/pt")

# 自动选设备
if torch.cuda.is_available():
    device, dtype = "cuda", torch.bfloat16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float32
else:
    device, dtype = "cpu", torch.float32

print(f"Loading model ({device})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map=device,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

print(f"\n{'='*50}")
print("  PT 模型对话（续写模式）")
print("  用自然语言提问，模型会续写回答")
print("  但它不懂对话规则——SFT后才能真正聊天")
print("  输入 quit 退出")
print(f"{'='*50}\n")

while True:
    try:
        user = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        break
    if user.lower() in ("quit", "exit", "q"):
        break
    if not user:
        continue

    # 包装成医患对话格式，引导模型续写
    prompt = f"患者问题：{user}\n医生回答："
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    print(f"Model: {response.strip()}\n")
