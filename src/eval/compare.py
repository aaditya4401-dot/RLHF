"""Pairwise preference evaluation using OpenAI API as automated judge.

Compares model pairs: Base vs SFT, SFT vs DPO, Base vs DPO.
For each pair, the judge picks a winner per prompt, then we compute
overall win rates to show the progression of improvement.

NOTE: This script runs locally — no GPU needed, just an OpenAI API key.
Set OPENAI_API_KEY in your .env file.

Usage:
    python -m src.eval.compare
    python -m src.eval.compare --results data/eval/results.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

# ── Defaults ────────────────────────────────────────────────────────────
RESULTS_FILE = "./data/eval/results.json"
OUTPUT_FILE = "./data/eval/preference_winrates.json"
JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

# The three pairwise comparisons we care about
COMPARISONS = [
    ("base_response", "sft_response", "Base", "SFT"),
    ("sft_response", "dpo_response", "SFT", "DPO"),
    ("base_response", "dpo_response", "Base", "DPO"),
]

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator comparing two AI-generated responses to a question.

Evaluate both responses on these criteria:
1. **Helpfulness** — Does it directly answer the question?
2. **Accuracy** — Is the information correct?
3. **Relevance** — Does it stay focused on what was asked?
4. **Conciseness** — Is it clear without filler?

You MUST respond with valid JSON only, no other text:
{"winner": "A" or "B", "reasoning": "Brief explanation."}"""

JUDGE_USER_TEMPLATE = """Question: {prompt}

--- Response A ---
{response_a}

--- Response B ---
{response_b}

Which response is better? Return JSON with "winner" and "reasoning"."""


def judge_pair(
    client: OpenAI,
    prompt: str,
    response_a: str,
    response_b: str,
    model: str = JUDGE_MODEL,
) -> str | None:
    """Ask the judge to pick a winner. Returns 'A', 'B', or None on error."""
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
            ),
        },
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        if "```" in raw:
            json_str = raw.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            verdict = json.loads(json_str.strip())
        else:
            return None

    return verdict.get("winner", "").upper()


def run_comparison(
    results: list[dict],
    key_a: str,
    key_b: str,
    label_a: str,
    label_b: str,
    client: OpenAI,
) -> dict:
    """Run pairwise comparison across all prompts for one model pair."""
    wins_a = 0
    wins_b = 0
    errors = 0

    for item in tqdm(results, desc=f"{label_a} vs {label_b}"):
        resp_a = item[key_a]
        resp_b = item[key_b]

        # Skip placeholder responses
        if "not available" in resp_a or "not available" in resp_b:
            errors += 1
            continue

        winner = judge_pair(client, item["prompt"], resp_a, resp_b)

        if winner == "A":
            wins_a += 1
        elif winner == "B":
            wins_b += 1
        else:
            errors += 1

    total_valid = wins_a + wins_b
    return {
        "comparison": f"{label_a} vs {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        f"{label_a}_wins": wins_a,
        f"{label_b}_wins": wins_b,
        "errors": errors,
        "total_judged": total_valid,
        f"{label_a}_win_rate": round(wins_a / total_valid * 100, 1) if total_valid > 0 else 0,
        f"{label_b}_win_rate": round(wins_b / total_valid * 100, 1) if total_valid > 0 else 0,
    }


def run_all_comparisons(
    results_file: str = RESULTS_FILE,
    output_file: str = OUTPUT_FILE,
) -> list[dict]:
    """Run all 3 pairwise comparisons and save results."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {results_file}")
    print(f"Judge model: {JUDGE_MODEL}\n")

    client = OpenAI()
    all_comparisons = []

    for key_a, key_b, label_a, label_b in COMPARISONS:
        # Skip comparisons involving missing model keys
        if key_a not in results[0] or key_b not in results[0]:
            print(f"  Skipping {label_a} vs {label_b} — '{key_a}' or '{key_b}' not in results")
            continue
        comp = run_comparison(results, key_a, key_b, label_a, label_b, client)
        all_comparisons.append(comp)

    # ── Save results ──
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    # ── Print summary ──
    print("\n" + "=" * 64)
    print("  PREFERENCE WIN RATES — Improvement Progression")
    print("=" * 64)

    for comp in all_comparisons:
        a = comp["model_a"]
        b = comp["model_b"]
        a_rate = comp[f"{a}_win_rate"]
        b_rate = comp[f"{b}_win_rate"]
        total = comp["total_judged"]

        # Visual bar (ASCII-safe for Windows)
        bar_len = 40
        a_bar = int(a_rate / 100 * bar_len)
        b_bar = bar_len - a_bar

        print(f"\n  {a} vs {b}  ({total} prompts judged)")
        print(f"  {a:<6} {a_rate:5.1f}% {'#' * a_bar}{'-' * b_bar} {b_rate:5.1f}% {b:>6}")

        if b_rate > a_rate:
            print(f"  → {b} wins ({b_rate - a_rate:.1f}pp improvement)")
        elif a_rate > b_rate:
            print(f"  → {a} wins ({a_rate - b_rate:.1f}pp)")
        else:
            print(f"  → Tie")

    print("\n" + "=" * 64)

    # Overall narrative
    if len(all_comparisons) >= 3:
        base_vs_dpo = all_comparisons[2]
        a = base_vs_dpo["model_a"]
        b = base_vs_dpo["model_b"]
        dpo_rate = base_vs_dpo[f"{b}_win_rate"]
        print(f"\n  End-to-end: DPO model preferred over Base in {dpo_rate}% of comparisons")
    elif all_comparisons:
        best = all_comparisons[0]
        a, b = best["model_a"], best["model_b"]
        b_rate = best[f"{b}_win_rate"]
        print(f"\n  {b} preferred over {a} in {b_rate}% of comparisons")
    print()

    return all_comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise preference evaluation")
    parser.add_argument("--results", default=RESULTS_FILE, help="Path to results.json from benchmark.py")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output win rates path")
    args = parser.parse_args()

    run_all_comparisons(results_file=args.results, output_file=args.output)
