#!/usr/bin/env python3
"""
train.py — QLoRA training script for Agent Soul.

Standard causal language model fine-tuning on clean conversation text.
Uses Unsloth for efficient QLoRA training with packing.

Input: JSONL file with {"text": "..."} per line (one session per line).
The model learns next-token prediction on conversation patterns.

Usage:
    python train.py --data train.jsonl --output ./adapter_out
    python train.py --data train.jsonl --base-adapter ./prev_adapter --output ./adapter_out
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ["TORCHDYNAMO_DISABLE"] = "1"

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _bf16_supported() -> bool:
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def load_model(base_model: str, base_adapter: str | None, rank: int, max_seq_length: int):
    """Load base model with Unsloth, optionally with existing LoRA adapter."""
    from unsloth import FastLanguageModel

    log.info("Loading base model: %s", base_model)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    if base_adapter:
        log.info("Loading existing adapter from: %s", base_adapter)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, base_adapter)
        log.info("Adapter loaded — continuing training")
    else:
        log.info("Creating new LoRA adapter (rank=%d)", rank)
        model = FastLanguageModel.get_peft_model(
            model,
            r=rank,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=rank * 2,
            lora_dropout=0.05,
            use_rslora=True,
            use_gradient_checkpointing="unsloth",
        )

    log.info("Model ready")
    return model, tokenizer


def load_dataset(data_path: Path):
    """Load plain text JSONL dataset."""
    from datasets import load_dataset as hf_load_dataset

    log.info("Loading dataset from: %s", data_path)

    dataset = hf_load_dataset(
        "json",
        data_files=str(data_path),
        split="train",
    )

    log.info("Dataset loaded — %d documents", len(dataset))

    # Verify format
    if "text" not in dataset.column_names:
        log.error("Dataset must have 'text' column. Found: %s", dataset.column_names)
        sys.exit(1)

    return dataset


def train(model, tokenizer, dataset, output_path: Path, epochs: int, max_seq_length: int):
    """Run causal LM training with Unsloth + SFTTrainer.

    Uses packing to efficiently combine multiple documents into sequences
    of max_seq_length. EOS tokens separate documents.
    """
    from trl import SFTTrainer, SFTConfig

    log.info("Configuring trainer — epochs=%d, max_seq=%d, output=%s",
             epochs, max_seq_length, output_path)

    # Ensure EOS token is set for document separation
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|endoftext|>"

    # Append EOS to each document for proper separation during packing
    def add_eos(examples):
        return {"text": [t + tokenizer.eos_token for t in examples["text"]]}

    dataset = dataset.map(add_eos, batched=True)

    training_args = SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="epoch",
        output_dir=str(output_path / "checkpoints"),
        fp16=not _bf16_supported(),
        bf16=_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    log.info("Starting training (%d documents, packing enabled)", len(dataset))
    train_start = datetime.now()

    stats = trainer.train()

    train_elapsed = (datetime.now() - train_start).total_seconds()
    log.info("Training complete in %.1fs", train_elapsed)
    log.info("Training loss: %.4f", stats.training_loss)

    return trainer


def save_adapter(model, tokenizer, output_path: Path, base_model: str):
    """Save LoRA adapter and tokenizer."""
    import json

    adapter_dir = output_path / "adapter"
    log.info("Saving adapter to: %s", adapter_dir)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Fix base_model_name_or_path for GGUF conversion
    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        original_base = cfg.get("base_model_name_or_path", "")
        if "bnb-4bit" in original_base or "unsloth" in original_base.lower():
            canonical = _resolve_canonical_model(base_model)
            cfg["base_model_name_or_path"] = canonical
            config_path.write_text(json.dumps(cfg, indent=2))
            log.info("Fixed base_model_name_or_path: %s -> %s", original_base, canonical)

    log.info("Adapter saved")
    return adapter_dir


_CANONICAL_MODELS = {
    "unsloth/Qwen3-8B": "Qwen/Qwen3-8B",
    "unsloth/Qwen3-4B": "Qwen/Qwen3-4B",
    "unsloth/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
    "unsloth/Qwen3-0.6B": "Qwen/Qwen3-0.6B",
}


def _resolve_canonical_model(base_model: str) -> str:
    if base_model in _CANONICAL_MODELS:
        return _CANONICAL_MODELS[base_model]
    if base_model.startswith("unsloth/"):
        name = base_model.removeprefix("unsloth/")
        for suffix in ["-unsloth-bnb-4bit", "-bnb-4bit", "-unsloth"]:
            name = name.removesuffix(suffix)
        return f"Qwen/{name}"
    return base_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QLoRA causal LM training for Agent Soul.",
    )
    parser.add_argument(
        "--data", "-d", type=Path, required=True,
        help="JSONL training data (each line: {\"text\": \"...\"})",
    )
    parser.add_argument(
        "--base-adapter", type=Path, default=None,
        help="Existing LoRA adapter to continue training from.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output directory for trained adapter.",
    )
    parser.add_argument(
        "--base-model", type=str, default="unsloth/Qwen3-8B",
        help="Base model (default: unsloth/Qwen3-8B).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1,
        help="Training epochs (default: 1).",
    )
    parser.add_argument(
        "--rank", type=int, default=16,
        help="LoRA rank (default: 16).",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Max sequence length for packing (default: 2048).",
    )
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Training data not found: %s", args.data)
        sys.exit(1)

    if args.base_adapter and not args.base_adapter.exists():
        log.error("Base adapter not found: %s", args.base_adapter)
        sys.exit(1)

    log.info("=" * 60)
    log.info("Agent Soul Training (Causal LM)")
    log.info("=" * 60)
    log.info("  Base model:      %s", args.base_model)
    log.info("  Training data:   %s", args.data)
    log.info("  Base adapter:    %s", args.base_adapter or "(none — fresh)")
    log.info("  Output:          %s", args.output)
    log.info("  Epochs:          %d", args.epochs)
    log.info("  LoRA rank:       %d", args.rank)
    log.info("  Max seq length:  %d", args.max_seq_length)
    log.info("=" * 60)

    pipeline_start = datetime.now()

    model, tokenizer = load_model(
        base_model=args.base_model,
        base_adapter=str(args.base_adapter) if args.base_adapter else None,
        rank=args.rank,
        max_seq_length=args.max_seq_length,
    )

    dataset = load_dataset(args.data)

    train(model, tokenizer, dataset, args.output, args.epochs, args.max_seq_length)

    adapter_dir = save_adapter(model, tokenizer, args.output, args.base_model)

    pipeline_elapsed = (datetime.now() - pipeline_start).total_seconds()
    log.info("=" * 60)
    log.info("Pipeline complete in %.1fs", pipeline_elapsed)
    log.info("Adapter: %s", adapter_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
