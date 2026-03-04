"""Compute automated metrics across all 3 model versions.

Metrics computed:
- ROUGE-L (measures overlap with chosen reference responses)
- Response length stats (word count mean/median/std)
- Perplexity is noted but requires GPU — compute it in benchmark.py if needed

NOTE: This script runs locally (no GPU required). It reads the results
produced by benchmark.py and the test set references.

Usage:
    python -m src.eval.metrics
    python -m src.eval.metrics --results data/eval/results.json --references data/eval/test.json
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from rouge_score import rouge_scorer


# ── Defaults ────────────────────────────────────────────────────────────
RESULTS_FILE = "./data/eval/results.json"
REFERENCES_FILE = "./data/eval/test.json"
OUTPUT_CSV = "./data/eval/metrics.csv"

MODEL_KEYS = [
    ("base_response", "Base Mistral 7B"),
    ("sft_response", "After LoRA SFT"),
    ("dpo_response", "After DPO"),
]


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_references(path: str) -> dict[str, str]:
    """Load test.json and build a prompt -> chosen response map."""
    refs = {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        refs[item["prompt"]] = item["chosen"]
    return refs


def compute_rouge_l(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score["rougeL"].fmeasure)
    return {
        "rouge_l_mean": statistics.mean(scores),
        "rouge_l_median": statistics.median(scores),
        "rouge_l_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def compute_length_stats(responses: list[str]) -> dict:
    """Compute word count statistics."""
    lengths = [len(r.split()) for r in responses]
    return {
        "word_count_mean": statistics.mean(lengths),
        "word_count_median": statistics.median(lengths),
        "word_count_std": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "word_count_min": min(lengths),
        "word_count_max": max(lengths),
    }


def compute_all_metrics(
    results_file: str = RESULTS_FILE,
    references_file: str = REFERENCES_FILE,
    output_csv: str = OUTPUT_CSV,
) -> list[dict]:
    """Compute metrics for all 3 model versions and save to CSV."""
    results = load_results(results_file)
    refs_map = load_references(references_file)

    # Build reference list aligned with results
    references = []
    for r in results:
        ref = refs_map.get(r["prompt"], "")
        references.append(ref)

    has_references = any(r != "" for r in references)

    all_metrics = []

    for key, label in MODEL_KEYS:
        # Skip if model key doesn't exist in results
        if key not in results[0]:
            metrics = {"model": label, "status": "not in results"}
            all_metrics.append(metrics)
            continue

        responses = [r[key] for r in results]

        # Skip placeholder responses
        if responses and responses[0].startswith("[") and "not available" in responses[0]:
            metrics = {"model": label, "status": "adapter not available"}
            all_metrics.append(metrics)
            continue

        metrics = {"model": label}

        # Length stats
        length = compute_length_stats(responses)
        metrics.update(length)

        # ROUGE-L (only if we have reference responses)
        if has_references:
            rouge = compute_rouge_l(responses, references)
            metrics.update(rouge)

        all_metrics.append(metrics)

    # ── Print summary table ──
    print("\n" + "=" * 80)
    print("EVALUATION METRICS COMPARISON")
    print("=" * 80)

    header_cols = ["Metric", *[m["model"] for m in all_metrics if "status" not in m]]
    active_metrics = [m for m in all_metrics if "status" not in m]

    if has_references:
        metric_rows = [
            ("ROUGE-L (mean)", "rouge_l_mean", ".3f"),
            ("ROUGE-L (median)", "rouge_l_median", ".3f"),
        ]
    else:
        metric_rows = []

    metric_rows.extend([
        ("Word count (mean)", "word_count_mean", ".1f"),
        ("Word count (median)", "word_count_median", ".1f"),
        ("Word count (std)", "word_count_std", ".1f"),
        ("Word count (min)", "word_count_min", ".0f"),
        ("Word count (max)", "word_count_max", ".0f"),
    ])

    # Print header
    col_width = 22
    print(f"\n{'Metric':<{col_width}}", end="")
    for m in active_metrics:
        print(f"  {m['model']:>{col_width}}", end="")
    print()
    print("-" * (col_width + (col_width + 2) * len(active_metrics)))

    # Print rows
    for display_name, key, fmt in metric_rows:
        print(f"{display_name:<{col_width}}", end="")
        for m in active_metrics:
            val = m.get(key, "N/A")
            if isinstance(val, (int, float)):
                print(f"  {val:>{col_width}{fmt}}", end="")
            else:
                print(f"  {val:>{col_width}}", end="")
        print()

    # Note unavailable models
    for m in all_metrics:
        if "status" in m:
            print(f"\n  [{m['model']}]: {m['status']}")

    print()

    # ── Save CSV ──
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten all metrics into CSV rows
    fieldnames = sorted(set().union(*(m.keys() for m in all_metrics)))
    fieldnames = ["model"] + [f for f in fieldnames if f != "model"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"Metrics saved to {output_path}")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute evaluation metrics")
    parser.add_argument("--results", default=RESULTS_FILE, help="Path to results.json from benchmark.py")
    parser.add_argument("--references", default=REFERENCES_FILE, help="Path to test.json with reference responses")
    parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
    args = parser.parse_args()

    compute_all_metrics(
        results_file=args.results,
        references_file=args.references,
        output_csv=args.output,
    )
