---
language:
- zh
tags:
- medical
- qwen
- lora
- chinese-medical
- doctor-song
- gguf
- quantized
pipeline_tag: text-generation
base_model: Qwen/Qwen3.5-0.8B-Base
---

# Doctor.Song-0.8B-Medical

中文医学对话大模型。基于 Qwen3.5-0.8B-Base，在单卡 RTX 3090 (24GB) 上完成 **PT → SFT → DPO → SFT-CoT** 四阶段全流程训练。

## 模型规格

| 项目 | 值 |
|------|-----|
| 基座模型 | Qwen3.5-0.8B-Base |
| 参数 | 0.8B (24层, hidden=1024, 8头注意力) |
| 词表大小 | 248,320 |
| 训练硬件 | 单卡 RTX 3090 (24GB) |
| 全流程微调耗时 | ~5 小时 |

## 可用版本

| 版本 | 路径 | 体积 | 用途 |
|------|------|------|------|
| FP32 | 根目录 | **2.82 GB** | 完整精度，适合 GPU 生产部署 |
| 4-bit NF4 | `/4bit` | **1.21 GB** | 双重量化，适合低显存/移动端 |
| GGUF Q4_K_M | `/gguf` | **505 MB** | llama.cpp 推理，Mac Metal 可达 63 tok/s |
| LoRA 适配器 | `/lora/*` | 40~60 MB/个 | 复现训练过程或继续微调 |

## 训练数据

| 阶段 | 样本量 | 方法 | 产物 |
|------|--------|------|------|
| PT 增量预训练 | 11,475 条 | 全参数 | 医学教科书/论文/临床指南语料 |
| SFT 指令微调 | 7,981 对 | LoRA (r=8, α=16) | `/lora/sft-lora` |
| DPO 偏好对齐 | 7,316 对 | LoRA (r=8, β=0.1) | `/lora/dpo-lora` |
| CoT 推理链 | 3,000 条 | LoRA (r=16) | `/lora/sft-cot-lora` |

---

## 使用方式

### Transformers（FP32 完整版）

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Songlei327/Doctor.Song-0.8B-Medical",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Songlei327/Doctor.Song-0.8B-Medical")

messages = [{"role": "user", "content": "感冒发烧了应该怎么办？"}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### 4-bit 量化版

```bash
pip install bitsandbytes
```

```python
model = AutoModelForCausalLM.from_pretrained(
    "Songlei327/Doctor.Song-0.8B-Medical",
    subfolder="4bit",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Songlei327/Doctor.Song-0.8B-Medical")
# 使用方式同上
```

### llama.cpp / GGUF

```bash
# 下载 GGUF
huggingface-cli download Songlei327/Doctor.Song-0.8B-Medical --local-dir . gguf/

# 直接推理
llama-cli -m gguf/doctor-song-Q4_K_M.gguf \
    -p "<|im_start|>user\n感冒发烧了应该怎么办？<|im_end|>\n<|im_start|>assistant\n" \
    -n 512 -t 4

# 或启动 OpenAI 兼容 API 服务
llama-server -m gguf/doctor-song-Q4_K_M.gguf \
    --port 8000 -ngl 99 --ctx-size 4096
# curl http://localhost:8000/v1/chat/completions
```

### LoRA 适配器（复现/继续训练）

```bash
# 合并 SFT LoRA 到基座模型
python tools/merge_peft_adapter.py \
    --base_model Qwen/Qwen3.5-0.8B-Base \
    --lora_model ./lora/sft-lora \
    --output_dir ./outputs/sft-merge

# 合并 DPO LoRA
python tools/merge_peft_adapter.py \
    --base_model ./outputs/sft-merge \
    --lora_model ./lora/dpo-lora \
    --output_dir ./outputs/dpo-merge
```

---

## 推理速度参考

| 环境 | 格式 | 速度 |
|------|------|------|
| RTX 3090 bf16 | FP32 | 预估 20-40 tok/s |
| Mac MPS (Apple Silicon) | FP32 | 实测 2.3 tok/s |
| Mac Metal + llama.cpp | GGUF Q4_K_M | 实测 ~63 tok/s |

---

## 免责声明

本模型仅限研究目的使用，**不能替代专业医生诊断**。模型输出不代表医学建议，具体医疗问题请咨询执业医师。
