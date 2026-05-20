# -*- coding: utf-8 -*-
"""简单 merge — CPU only, 低内存"""
import os, glob, shutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "outputs-pt-doctor-song"
LORA = "outputs-sft-doctor-song-lora"
OUT = "outputs-sft-doctor-song-merged"

print("Loading base model (CPU, low memory)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA, device_map="cpu")

print("Merging...")
model = model.merge_and_unload()

os.makedirs(OUT, exist_ok=True)
print("Saving merged model...")
model.save_pretrained(OUT, max_shard_size="10GB")

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
tokenizer.save_pretrained(OUT)

# 补全 tokenizer 文件
import glob, shutil
for pat in ["tokenizer_config.json", "tokenizer.json", "chat_template.jinja",
            "special_tokens_map.json", "generation_config.json", "vocab.json", "merges.txt"]:
    src = BASE + "/" + pat
    if os.path.exists(src):
        shutil.copy2(src, OUT + "/" + pat)

print(f"Done! Merged model at: {OUT}")
