"""Convert judged preference pairs into Hugging Face Datasets format."""

import json
from pathlib import Path

from datasets import Dataset, DatasetDict


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "preference_data"
EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"
TEST_RATIO = 0.1
SEED = 42


def load_judgments(path: str | Path | None = None) -> list[dict]:
    """Load judged preference pairs from JSON."""
    if path is None:
        path = DATA_DIR / "judgments.json"
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} preference pairs from {path}")
    return pairs


def pairs_to_dataset(pairs: list[dict]) -> Dataset:
    """Convert preference pairs list into a Hugging Face Dataset.

    Keeps only the columns needed for DPO training: prompt, chosen, rejected.
    """
    records = [
        {
            "prompt": p["prompt"],
            "chosen": p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ]
    return Dataset.from_list(records)


def split_dataset(dataset: Dataset, test_ratio: float = TEST_RATIO, seed: int = SEED) -> DatasetDict:
    """Split into train/test sets."""
    split = dataset.train_test_split(test_size=test_ratio, seed=seed)
    print(f"Train: {len(split['train'])} examples, Test: {len(split['test'])} examples")
    return split


def save_dataset(dataset_dict: DatasetDict, output_dir: str | Path | None = None, eval_dir: str | Path | None = None):
    """Save train split to preference_data/ and test split to eval/."""
    if output_dir is None:
        output_dir = DATA_DIR
    if eval_dir is None:
        eval_dir = EVAL_DIR
    output_dir = Path(output_dir)
    eval_dir = Path(eval_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    train_ds = dataset_dict["train"]
    test_ds = dataset_dict["test"]

    # Save train split — JSON + Parquet
    train_ds.to_json(output_dir / "train.json")
    train_ds.to_parquet(output_dir / "train.parquet")
    print(f"Saved train set ({len(train_ds)} rows) to {output_dir}")

    # Save test split — JSON array + Parquet into eval/ (held out from training)
    with open(eval_dir / "test.json", "w", encoding="utf-8") as _f:
        json.dump([row for row in test_ds], _f, indent=2, ensure_ascii=False)
    test_ds.to_parquet(eval_dir / "test.parquet")
    print(f"Saved test set ({len(test_ds)} rows) to {eval_dir}")

    # Also save the full HF DatasetDict for convenience
    dataset_dict.save_to_disk(output_dir / "hf_dataset")
    print(f"Saved HF DatasetDict to {output_dir / 'hf_dataset'}")


def format_and_save(judgments_path: str | Path | None = None) -> DatasetDict:
    """Full pipeline: load judgments, convert, split, save."""
    pairs = load_judgments(judgments_path)
    dataset = pairs_to_dataset(pairs)
    dataset_dict = split_dataset(dataset)
    save_dataset(dataset_dict)
    return dataset_dict


if __name__ == "__main__":
    ds = format_and_save()

    # Preview
    print("\n--- Train sample ---")
    sample = ds["train"][0]
    print(f"Prompt:   {sample['prompt']}")
    print(f"Chosen:   {sample['chosen'][:120]}...")
    print(f"Rejected: {sample['rejected'][:120]}...")

    print(f"\n--- Test sample ---")
    sample = ds["test"][0]
    print(f"Prompt:   {sample['prompt']}")
    print(f"Chosen:   {sample['chosen'][:120]}...")
    print(f"Rejected: {sample['rejected'][:120]}...")
