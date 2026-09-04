"""Stage 2: DPO training on the LoRA-tuned model.

Script form of `notebooks/kaggle_dpo.ipynb`. Unlike SFT, DPO uses both the
`chosen` and `rejected` columns — it optimizes the log-probability *margin*
between them, relative to a frozen reference policy.

The memory trick that makes this fit in 16 GB:

    1. Load the 4-bit base model.
    2. Load the SFT adapter and `merge_and_unload()` it into the weights.
       The model now *is* the SFT model, with no adapter attached.
    3. Attach a FRESH LoRA for DPO and pass `ref_model=None`.

TRL then derives the reference policy by temporarily disabling the adapter,
so adapter-off == merged SFT model. That gives DPO the two policies it needs
(pi_theta and pi_ref) from a single set of weights instead of two 7B models.

Merging in step 2 is what makes the reference correct: if the SFT adapter
were merely trained further instead, adapter-off would yield the *base* model
and the KL penalty would pull the policy back toward un-fine-tuned Mistral,
undoing stage 1.

Requires a GPU with bitsandbytes support. Hyperparameters come from
`src/training/config.py`; CLI flags override them.

Usage:
    python -m src.training.dpo_train --dataset data/preference_data/train.json \
        --sft-adapter models/sft_adapter
    python -m src.training.dpo_train --dataset train.json --sft-adapter ./sft_adapter --beta 0.3
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

from src.training import config


def build_quant_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config for the frozen base model."""
    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_compute_dtype=getattr(torch, config.BNB_4BIT_COMPUTE_DTYPE),
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
    )


def load_merged_sft_model(sft_adapter_dir: str | Path, base_model: str = config.BASE_MODEL):
    """Load the base model, merge the SFT adapter in, and prepare for QLoRA.

    The returned model is the SFT policy with no adapter attached — it becomes
    the DPO reference once a fresh adapter is layered on top.
    """
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=build_quant_config(),
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-padding is standard for DPO

    sft_model = PeftModel.from_pretrained(base, str(sft_adapter_dir), is_trainable=False)
    print(f"SFT adapter loaded from: {sft_adapter_dir}")

    model = sft_model.merge_and_unload()
    print("SFT adapter merged into base weights — this is now the reference policy.")

    model = prepare_model_for_kbit_training(model)
    print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer


def load_preference_dataset(dataset_path: str | Path):
    """Load preference pairs, keeping the prompt/chosen/rejected triple."""
    ds = load_dataset("json", data_files=str(dataset_path), split="train")

    keep = ("prompt", "chosen", "rejected")
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    print(f"Loaded {len(ds)} preference pairs with columns {ds.column_names}")
    return ds


def build_dpo_lora_config() -> LoraConfig:
    """Fresh LoRA adapter for DPO — deliberately not the SFT adapter."""
    return LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
        task_type=config.LORA_TASK_TYPE,
    )


def build_dpo_args(output_dir: str | Path, beta: float, fp16: bool) -> DPOConfig:
    """Assemble DPOConfig from config.

    beta is the KL leash: too low and the policy drifts into degenerate output
    (0.1 diverged on this dataset), too high and it barely moves (0.5 was
    too conservative). 0.3 was the working value.
    """
    return DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=config.DPO_EPOCHS,
        per_device_train_batch_size=config.DPO_BATCH_SIZE,
        gradient_accumulation_steps=config.DPO_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.DPO_LEARNING_RATE,
        warmup_steps=config.DPO_WARMUP_STEPS,
        lr_scheduler_type=config.DPO_LR_SCHEDULER,
        weight_decay=config.DPO_WEIGHT_DECAY,
        fp16=fp16,
        bf16=False,
        gradient_checkpointing=config.DPO_GRADIENT_CHECKPOINTING,
        logging_steps=config.DPO_LOGGING_STEPS,
        save_steps=config.DPO_SAVE_STEPS,
        save_total_limit=3,
        report_to="none",
        run_name=config.WANDB_RUN_NAME_DPO,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        beta=beta,
        max_length=config.DPO_MAX_LENGTH,
        max_prompt_length=config.DPO_MAX_PROMPT_LENGTH,
    )


def build_trainer(model, tokenizer, dataset, training_args):
    """Construct the DPOTrainer, tolerating the TRL API rename.

    `processing_class` replaced `tokenizer` in newer TRL; Kaggle's pinned
    version varies, so try the current signature and fall back.
    """
    kwargs = dict(
        model=model,
        ref_model=None,  # adapter-disabled model == merged SFT == reference policy
        args=training_args,
        train_dataset=dataset,
        peft_config=build_dpo_lora_config(),
    )

    try:
        return DPOTrainer(processing_class=tokenizer, **kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **kwargs)


def train(
    dataset_path: str | Path,
    sft_adapter_dir: str | Path,
    adapter_dir: str | Path = config.DPO_ADAPTER_DIR,
    output_dir: str | Path = config.DPO_OUTPUT_DIR,
    beta: float = config.DPO_BETA,
    resume_from_checkpoint: str | None = None,
    fp16: bool = config.DPO_FP16,
):
    """Run the full DPO stage and save the adapter."""
    model, tokenizer = load_merged_sft_model(sft_adapter_dir)
    dataset = load_preference_dataset(dataset_path)

    trainer = build_trainer(model, tokenizer, dataset, build_dpo_args(output_dir, beta, fp16))

    effective_batch = config.DPO_BATCH_SIZE * config.DPO_GRADIENT_ACCUMULATION_STEPS
    print(f"\nDPO beta: {beta}")
    print(f"Learning rate: {config.DPO_LEARNING_RATE}")
    print(f"Training samples: {len(dataset)}")
    print(f"Effective batch size: {effective_batch}")
    print(f"Epochs: {config.DPO_EPOCHS}")
    trainer.model.print_trainable_parameters()

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    adapter_dir = Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"\nDPO adapter saved to {adapter_dir}")
    for f in sorted(os.listdir(adapter_dir)):
        size = os.path.getsize(adapter_dir / f)
        print(f"  {f}: {size / 1e6:.1f} MB")

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: DPO preference alignment")
    parser.add_argument(
        "--dataset",
        default="data/preference_data/train.json",
        help="Preference pairs JSON/JSONL with prompt/chosen/rejected columns",
    )
    parser.add_argument(
        "--sft-adapter",
        default=config.SFT_ADAPTER_DIR,
        help="SFT adapter to merge in before training",
    )
    parser.add_argument("--adapter-dir", default=config.DPO_ADAPTER_DIR, help="Where to save the DPO adapter")
    parser.add_argument("--output-dir", default=config.DPO_OUTPUT_DIR, help="Checkpoint directory")
    parser.add_argument("--beta", type=float, default=config.DPO_BETA, help="KL penalty strength")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=config.DPO_FP16,
        help="Enable fp16 (off by default — the grad scaler conflicts with 4-bit on P100)",
    )
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        sft_adapter_dir=args.sft_adapter,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        beta=args.beta,
        resume_from_checkpoint=args.resume,
        fp16=args.fp16,
    )
