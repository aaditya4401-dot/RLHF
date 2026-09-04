"""Stage 1: LoRA Supervised Fine-Tuning on Mistral 7B.

Script form of `notebooks/kaggle_sft.ipynb`. Loads the base model in 4-bit
NF4, attaches a LoRA adapter to the attention projections, and trains on the
`chosen` column of the preference dataset (SFT only imitates good responses —
the `rejected` column is used later by DPO).

Trains ~13.6M parameters out of 7.26B (0.19%), producing a ~27 MB adapter.

Requires a GPU with bitsandbytes support. Hyperparameters come from
`src/training/config.py`; CLI flags override them.

Usage:
    python -m src.training.sft_lora --dataset data/preference_data/train.json
    python -m src.training.sft_lora --dataset train.json --adapter-dir ./sft_adapter
    python -m src.training.sft_lora --dataset train.json --resume ./sft_output/checkpoint-500
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from src.training import config


def build_quant_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for the frozen base model.

    compute_dtype is float16 rather than bfloat16 — the Tesla P100 these runs
    targeted is Pascal-era and has no bf16 support.
    """
    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )


def load_base_model(base_model: str = config.BASE_MODEL):
    """Load the quantized base model and tokenizer, prepared for QLoRA."""
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_quant_config(),
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Casts layer norms to fp32 and enables gradient checkpointing so
    # gradients can flow through the 4-bit frozen weights to the adapters.
    model = prepare_model_for_kbit_training(model)

    print(f"Model loaded: {base_model}")
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer


def attach_lora(model):
    """Attach a fresh LoRA adapter to the attention projections."""
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_sft_dataset(dataset_path: str | Path):
    """Load preference pairs and keep only the prompt/completion columns.

    SFT trains on `chosen` alone, so `rejected` is dropped here. TRL's
    prompt-completion format applies the model's chat template itself, which
    is why no manual `[INST] ... [/INST]` string is built.
    """
    ds = load_dataset("json", data_files=str(dataset_path), split="train")
    print(f"Loaded {len(ds)} examples with columns {ds.column_names}")

    ds = ds.rename_column("chosen", "completion")
    drop = [c for c in ds.column_names if c not in ("prompt", "completion")]
    if drop:
        ds = ds.remove_columns(drop)

    print(f"SFT columns: {ds.column_names}")
    return ds


def build_training_args(output_dir: str | Path, fp16: bool) -> TrainingArguments:
    """Assemble TrainingArguments from config."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.SFT_EPOCHS,
        per_device_train_batch_size=config.SFT_BATCH_SIZE,
        gradient_accumulation_steps=config.SFT_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.SFT_LEARNING_RATE,
        warmup_steps=config.SFT_WARMUP_STEPS,
        lr_scheduler_type=config.SFT_LR_SCHEDULER,
        weight_decay=config.SFT_WEIGHT_DECAY,
        fp16=fp16,
        bf16=False,
        gradient_checkpointing=config.SFT_GRADIENT_CHECKPOINTING,
        logging_steps=config.SFT_LOGGING_STEPS,
        save_steps=config.SFT_SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        run_name=config.WANDB_RUN_NAME_SFT,
        optim="paged_adamw_8bit",
    )


def train(
    dataset_path: str | Path,
    adapter_dir: str | Path = config.SFT_ADAPTER_DIR,
    output_dir: str | Path = config.SFT_OUTPUT_DIR,
    resume_from_checkpoint: str | None = None,
    fp16: bool = config.SFT_FP16,
):
    """Run the full SFT stage and save the adapter."""
    model, tokenizer = load_base_model()
    model = attach_lora(model)
    sft_ds = load_sft_dataset(dataset_path)

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_ds,
        args=build_training_args(output_dir, fp16),
        processing_class=tokenizer,
    )

    effective_batch = config.SFT_BATCH_SIZE * config.SFT_GRADIENT_ACCUMULATION_STEPS
    print(f"\nTraining samples: {len(sft_ds)}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Epochs: {config.SFT_EPOCHS}")
    if resume_from_checkpoint:
        print(f"Resuming from: {resume_from_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"\nSFT adapter saved to {adapter_dir}")
    for f in sorted(os.listdir(adapter_dir)):
        size = os.path.getsize(adapter_dir / f)
        print(f"  {f}: {size / 1e6:.1f} MB")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: QLoRA supervised fine-tuning")
    parser.add_argument(
        "--dataset",
        default="data/preference_data/train.json",
        help="Preference pairs JSON/JSONL with prompt/chosen/rejected columns",
    )
    parser.add_argument("--adapter-dir", default=config.SFT_ADAPTER_DIR, help="Where to save the LoRA adapter")
    parser.add_argument("--output-dir", default=config.SFT_OUTPUT_DIR, help="Checkpoint directory")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=config.SFT_FP16,
        help="Enable fp16 (off by default — the grad scaler conflicts with 4-bit on P100)",
    )
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume,
        fp16=args.fp16,
    )
