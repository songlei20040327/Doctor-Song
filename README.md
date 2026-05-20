# Doctor.Song — 中文医学大模型

基于 Qwen3.5-0.8B-Base，在**单卡 RTX 3090 (24GB)** 上完成 PT → SFT → DPO → CoT 四阶段全流程训练的中文医学对话模型。

> **🤗 HuggingFace**: [Songlei327/Doctor.Song-0.8B-Medical](https://huggingface.co/Songlei327/Doctor.Song-0.8B-Medical)

## 核心指标

| 指标 | 数值 | 说明 |
|------|------|------|
| LLM-as-Judge 综合 | **3.10 / 5** | DeepSeek-chat 医学专家评判 |
| 安全性 | **4.0 / 5** | 专业级免责声明和安全提示 |
| Perplexity | **3.92** | 医学问答 holdout 集 |
| 模型体积 | **2.82 GB** (FP32) / **1.21 GB** (4-bit) | 可直接部署消费级设备 |
| 全流程微调耗时 | **~5 小时** | 单卡 RTX 3090 |

## 模型下载

```bash
# 下载完整模型 (FP32, 2.82GB)
huggingface-cli download Songlei327/Doctor.Song-0.8B-Medical --local-dir ./model

# 或只下载 4-bit 量化版 (1.21GB)
huggingface-cli download Songlei327/Doctor.Song-0.8B-Medical --local-dir ./model --include "4bit/*"

# 或只下载 GGUF (505MB)
huggingface-cli download Songlei327/Doctor.Song-0.8B-Medical --local-dir ./model --include "gguf/*"
```

HF 仓库包含：FP32 完整模型 / 4-bit 量化版 / GGUF 格式 / LoRA 适配器。

## 快速开始

### 训练

```bash
pip install -r requirements.txt

bash scripts/run_pt_3090.sh         # 增量预训练
bash scripts/run_sft_lora_3090.sh   # SFT 微调 (LoRA, 79分钟)
bash scripts/run_dpo_3090.sh        # DPO 对齐 (LoRA, 118分钟)
bash scripts/run_sft_cot_3090.sh    # CoT 推理链注入 (LoRA, 104分钟)
```

### 部署

```bash
# 方式 1：OpenAI 兼容 API
bash scripts/run_api_3090.sh        # GPU
bash scripts/run_api_local.sh       # Mac (MPS)
# → http://localhost:8000/docs

# 方式 2：llama.cpp GGUF（Mac 推荐，63 tok/s）
bash scripts/run_gguf_server.sh
# → http://localhost:8000/v1/chat/completions

# 方式 3：Vue 3 Web UI
cd frontend && npm install && npm run dev
# → http://localhost:5173（深色模式 + 对话/知识库/笔记）

# 方式 4：Gradio
bash scripts/run_gradio_3090.sh
# → http://localhost:8081
```

### 合并 & 量化

```bash
# LoRA → 完整模型
python tools/merge_peft_adapter.py \
    --base_model ./outputs/pt --lora_model ./outputs/sft-lora \
    --output_dir ./outputs/sft-merge

# 4-bit 量化
python tools/model_quant.py \
    --model_path ./outputs/final --output_dir ./outputs/final-4bit --bits 4
```

## 训练流水线

```
PT (领域预训练) → SFT (指令微调) → DPO (安全对齐) → SFT-CoT (推理链注入)
```

| 阶段 | 数据量 | 方法 | 耗时 | 关键指标 |
|------|--------|------|------|----------|
| PT | 11,475 条 | 全参数 | ~3h | 覆盖 4 大医学领域 |
| SFT | 7,981 对 | LoRA r=8 | **79 min** | PPL **3.92** |
| DPO | 7,316 对 | LoRA r=8, β=0.1 | **118 min** | Eval Loss **0.61** |
| CoT | 3,000 条 | LoRA r=16 | **104 min** | DeepSeek 蒸馏推理链 |

## LLM-as-Judge 评测

DeepSeek-chat 评判，20 道医学题，5 分制：

| 模型 | 准确性 | 推理质量 | 安全性 | **综合** |
|------|--------|----------|--------|----------|
| **SFT** | 2.80 | 2.60 | **4.00** | **3.10** |
| DPO | 2.40 | 2.20 | 3.40 | 2.64 |
| CoT | 2.40 | 2.20 | 3.20 | 2.58 |

- 安全分 4.0/5，最佳单题 4.3/5（老年人症状咨询）
- 药理知识是 0.8B 模型的共同短板

## 推理速度

| 环境 | 格式 | 速度 |
|------|------|------|
| RTX 3090 bf16 | FP32 | 预估 20-40 tok/s |
| Mac MPS | FP32 | 实测 2.3 tok/s |
| Mac Metal + llama.cpp | GGUF Q4_K_M | 实测 **~63 tok/s** |

## 项目结构

```
MedicalGPT/
├── training/                     # 核心训练脚本
│   ├── utils.py                     # 共享工具（LoRA/量化/数据加载）
│   ├── template.py                  # 30+ 模型对话模板
│   ├── tool_utils.py                # Agent/Function Call 工具
│   ├── supervised_finetuning.py     # SFT
│   ├── dpo_training.py              # DPO
│   ├── pretraining.py               # PT 预训练
│   ├── grpo_training.py             # GRPO 强化学习
│   ├── orpo_training.py             # ORPO
│   ├── ppo_training.py              # PPO/RLOO
│   ├── reward_modeling.py           # 奖励模型
│   └── opd_training.py              # On-Policy 蒸馏
│
├── scripts/                      # 一键运行脚本
│   ├── run_pt_3090.sh
│   ├── run_sft_lora_3090.sh / run_sft_full_3090.sh
│   ├── run_dpo_3090.sh / run_dpo_cloud.sh
│   ├── run_sft_cot_3090.sh
│   ├── run_grpo_3090.sh
│   ├── run_api_3090.sh / run_api_local.sh
│   ├── run_gguf_server.sh / run_gradio_3090.sh
│   ├── upload_to_hf.sh / upload_to_hf.py
│   └── hf_model_card.md
│
├── demo/                         # 推理与部署
│   ├── openai_api.py                # OpenAI 兼容 API 服务
│   ├── gradio_demo.py               # Gradio Web UI
│   ├── inference.py                 # 命令行推理
│   └── web_ui.html                  # 独立 HTML 前端
│
├── frontend/                     # Vue 3 前端
│   └── src/views/
│       ├── LoginPage.vue            # 海绵宝宝主题登录页
│       ├── MainPage.vue             # 侧边栏 + 深色模式
│       └── chat/ChatPage.vue        # 核心对话页面
│
├── tools/                        # 工具集
│   ├── merge_peft_adapter.py        # LoRA 权重合并
│   ├── merge_simple.py              # 轻量合并 (CPU)
│   ├── model_quant.py               # 模型量化
│   ├── prepare_data.py / prepare_dpo_data.py
│   ├── generate_cot.py              # CoT 蒸馏生成
│   ├── eval_quality.py / compare_models.py
│   ├── convert_dataset.py / validate_jsonl.py
│   └── test_pt_model.py / chat_pt.py
│
├── data/                         # 训练数据
├── outputs/                      # 训练产出 (gitignored)
├── eval_results/                 # 评测结果
└── tests/                        # 单元测试
```

## 支持的模型

| 系列 | 规模 | Target Modules |
|------|------|---------------|
| Qwen3.5 | 0.8B-122B | q_proj, v_proj |
| Qwen3 | 0.6B-235B | q_proj, v_proj |
| Qwen2.5/2 | 0.5B-72B | q_proj, v_proj |
| LLaMA3 | 8B/70B | q_proj, v_proj |
| DeepSeek3 | 671B | q_proj, v_proj |
| ChatGLM3 | 6B | query_key_value |
| Mistral | 7B/8x7B | q_proj, v_proj |

完整模板名列表见 `training/template.py`。

## Agent 工具调用

数据中混入 tool call 格式，训练时加 `--tool_format` 参数即可。支持 `default` / `qwen` / `glm4` / `llama3` / `mistral`。

## License

- 代码：[Apache License 2.0](LICENSE)
- 模型权重：仅限研究目的，详见 [DISCLAIMER](DISCLAIMER)

## 致谢

基于 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)。项目源码：[songlei20040327/Doctor-Song](https://github.com/songlei20040327/Doctor-Song)。

参考：DPO ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) · ORPO ([arXiv:2403.07691](https://arxiv.org/abs/2403.07691)) · GRPO ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
