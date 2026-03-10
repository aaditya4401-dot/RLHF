"""Run all 3 model versions (base, SFT, DPO) and generate responses.

NOTE: This script requires a GPU. Run it on Kaggle or a machine with CUDA.
It loads Mistral 7B in 4-bit quantization (~5 GB VRAM) and generates
responses from each model version for every prompt in the held-out test set.

Usage:
    python -m src.eval.benchmark
    python -m src.eval.benchmark --test-file data/eval/test.json
    python -m src.eval.benchmark --sft-adapter models/sft_adapter --dpo-adapter models/dpo_adapter
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm


# ── Defaults ────────────────────────────────────────────────────────────
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
SFT_ADAPTER_DIR = "./models/sft_adapter"
DPO_ADAPTER_DIR = "./models/dpo_adapter"
TEST_FILE = "./data/eval/test.json"
OUTPUT_FILE = "./data/eval/results.json"
MAX_NEW_TOKENS = 300


def load_test_prompts(path: str) -> list[str]:
    """Load prompts from the held-out test set."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = [item["prompt"] for item in data]
    print(f"Loaded {len(prompts)} test prompts from {path}")
    return prompts


def get_bnb_config() -> BitsAndBytesConfig:
    """Standard 4-bit NF4 quantization config."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(model_name: str = BASE_MODEL):
    """Load the base Mistral 7B model in 4-bit."""
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def load_adapter_model(base_model, adapter_dir: str, label: str):
    """Load a LoRA adapter on top of a base model."""
    adapter_path = Path(adapter_dir)
    if not adapter_path.exists():
        print(f"  WARNING: {label} adapter not found at {adapter_dir}, skipping")
        return None
    print(f"  Loading {label} adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate a single response using the Mistral instruct format."""
    formatted = f"[INST] {prompt} [/INST]"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the generated part (skip the prompt tokens)
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_benchmark(
    test_file: str = TEST_FILE,
    sft_adapter_dir: str = SFT_ADAPTER_DIR,
    dpo_adapter_dir: str = DPO_ADAPTER_DIR,
    output_file: str = OUTPUT_FILE,
):
    """Run all 3 model versions on the test set."""
    prompts = load_test_prompts(test_file)

    # Load base model (shared across all versions)
    base_model, tokenizer = load_base_model()
    base_model.eval()

    # Generate base responses
    print("\n--- Generating base model responses ---")
    base_responses = []
    for prompt in tqdm(prompts, desc="Base"):
        base_responses.append(generate_response(base_model, tokenizer, prompt))

    # Load SFT adapter and generate
    sft_model = load_adapter_model(base_model, sft_adapter_dir, "SFT")
    sft_responses = []
    if sft_model is not None:
        print("\n--- Generating SFT model responses ---")
        for prompt in tqdm(prompts, desc="SFT"):
            sft_responses.append(generate_response(sft_model, tokenizer, prompt))
        del sft_model
        torch.cuda.empty_cache()
    else:
        sft_responses = ["[SFT adapter not available]"] * len(prompts)

    # Load DPO adapter and generate
    # The DPO adapter was trained on SFT-merged weights (not raw base).
    # We must merge the SFT adapter into the base first, then load the DPO adapter.
    dpo_adapter_path = Path(dpo_adapter_dir)
    sft_adapter_path = Path(sft_adapter_dir)
    dpo_responses = []

    if dpo_adapter_path.exists():
        print("\n--- Preparing DPO model (merge SFT + load DPO adapter) ---")

        if sft_adapter_path.exists():
            # Reload base for DPO (the previous sft_model may have been deleted)
            dpo_base, _ = load_base_model()
            sft_for_merge = PeftModel.from_pretrained(dpo_base, sft_adapter_dir)
            merged_model = sft_for_merge.merge_and_unload()
            del sft_for_merge
            torch.cuda.empty_cache()

            # Now load DPO adapter on top of merged SFT model
            dpo_model = PeftModel.from_pretrained(merged_model, dpo_adapter_dir)
            dpo_model.eval()
        else:
            # No SFT adapter — try loading DPO directly on base (may not work correctly)
            print("  WARNING: SFT adapter not found, loading DPO on raw base")
            dpo_model = load_adapter_model(base_model, dpo_adapter_dir, "DPO")

        if dpo_model is not None:
            print("\n--- Generating DPO model responses ---")
            for prompt in tqdm(prompts, desc="DPO"):
                dpo_responses.append(generate_response(dpo_model, tokenizer, prompt))
            del dpo_model
            torch.cuda.empty_cache()
        else:
            dpo_responses = ["[DPO adapter not available]"] * len(prompts)
    else:
        print(f"  WARNING: DPO adapter not found at {dpo_adapter_dir}, skipping")
        dpo_responses = ["[DPO adapter not available]"] * len(prompts)

    # Assemble results
    results = []
    for i, prompt in enumerate(prompts):
        results.append({
            "prompt": prompt,
            "base_response": base_responses[i],
            "sft_response": sft_responses[i],
            "dpo_response": dpo_responses[i],
        })

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} results to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark all 3 model versions")
    parser.add_argument("--test-file", default=TEST_FILE, help="Path to test.json")
    parser.add_argument("--sft-adapter", default=SFT_ADAPTER_DIR, help="SFT adapter dir")
    parser.add_argument("--dpo-adapter", default=DPO_ADAPTER_DIR, help="DPO adapter dir")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output results path")
    args = parser.parse_args()

    results = run_benchmark(
        test_file=args.test_file,
        sft_adapter_dir=args.sft_adapter,
        dpo_adapter_dir=args.dpo_adapter,
        output_file=args.output,
    )

    # Preview first result
    if results:
        r = results[0]
        print(f"\n{'='*60}")
        print(f"Prompt: {r['prompt']}")
        print(f"\n[Base]  {r['base_response'][:150]}...")
        print(f"\n[SFT]   {r['sft_response'][:150]}...")
        print(f"\n[DPO]   {r['dpo_response'][:150]}...")
