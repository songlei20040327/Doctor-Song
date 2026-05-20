"""Shared utilities for all training scripts."""
import os
import glob
import torch
from loguru import logger
from transformers import BitsAndBytesConfig, Trainer
from transformers.trainer import TRAINING_ARGS_NAME


class SavePeftModelTrainer(Trainer):
    """Trainer that saves LoRA adapter weights."""

    def save_model(self, output_dir=None, _internal_call=False):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        self.model.save_pretrained(output_dir)


def save_model(model, tokenizer, output_dir):
    """Save model and tokenizer. Handles distributed/parallel training."""
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, output_dir, trainer):
    """Save model for DeepSpeed ZeRO-3 via consolidated state dict."""
    os.makedirs(output_dir, exist_ok=True)
    state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir, state_dict=state_dict)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """Log counts of trainable vs. all parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"trainable params: {trainable} || all params: {total} "
        f"|| trainable%: {100 * trainable / total:.2f}"
    )


def find_all_linear_names(model, int4=False, int8=False):
    """Return sorted unique linear layer names for LoRA target_modules."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, cls):
            continue
        if 'lm_head' in name or 'output_layer' in name or 'score' in name:
            continue
        parts = name.split('.')
        names.add(parts[0] if len(parts) == 1 else parts[-1])
    return sorted(names)


def setup_tokenizer(tokenizer, template=None):
    """Set fallback eos/bos/pad tokens when they are missing."""
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = template.stop_str if template else "</s>"
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
        tokenizer.add_special_tokens({"pad_token": tokenizer.pad_token})
    return tokenizer


def build_quantization_config(load_in_4bit, load_in_8bit, torch_dtype, qlora=False):
    """Build BitsAndBytesConfig or raise on incompatibilities."""
    if load_in_4bit and load_in_8bit:
        raise ValueError("load_in_4bit and load_in_8bit cannot be set at the same time")
    if not (load_in_4bit or load_in_8bit):
        return None

    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)

    if qlora:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
    )


def load_jsonl_datasets(data_dir, validation_split_percentage=0.1, max_samples=None):
    """Load train/valid datasets from a directory of .jsonl files.

    If `data_dir` looks like a HuggingFace dataset ID (contains '/'),
    it will be loaded from the hub instead.
    """
    from datasets import load_dataset, DatasetDict

    if data_dir is not None and "/" in data_dir and " " not in data_dir and not os.path.isdir(data_dir):
        datasets = load_dataset(data_dir)
        if validation_split_percentage > 0:
            if "validation" not in datasets:
                split = datasets["train"].train_test_split(
                    test_size=validation_split_percentage, seed=42
                )
                datasets = DatasetDict({"train": split["train"], "validation": split["test"]})
            elif "test" in datasets:
                datasets["validation"] = datasets.pop("test")
        return datasets

    files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    logger.info(f"Loading {len(files)} data files from {data_dir}")

    datasets = load_dataset("json", data_files=files, split="train")
    if max_samples and max_samples > 0:
        datasets = datasets.select(range(min(max_samples, len(datasets))))

    if validation_split_percentage > 0:
        split = datasets.train_test_split(test_size=validation_split_percentage, seed=42)
        datasets = DatasetDict({"train": split["train"], "validation": split["test"]})
    else:
        datasets = DatasetDict({"train": datasets})
    return datasets
