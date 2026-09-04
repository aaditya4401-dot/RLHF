"""Pairwise preference evaluation using OpenAI API as automated judge.

Compares model pairs: Base vs SFT, SFT vs DPO, Base vs DPO.
For each pair, the judge picks a winner per prompt, then we compute
overall win rates to show the progression of improvement.

NOTE: This script runs locally — no GPU needed, just an OpenAI API key.
Set OPENAI_API_KEY in your .env file.

Win rates are reported with 95% Wilson score intervals and an exact binomial
test against chance (50%), so a result can be read as significant or not
rather than taken at face value. At n=198 the interval is roughly +/-7pp,
which is wide enough to matter.

Usage:
    python -m src.eval.compare
    python -m src.eval.compare --results data/eval/results.json
    python -m src.eval.compare --recompute    # re-derive stats, no API calls
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # openai is only needed on the judging path
    from openai import OpenAI


def _load_api_client():
    """Import and construct the OpenAI client on demand.

    Kept lazy so `--recompute`, which only does arithmetic on existing
    judgments, runs without the API dependencies installed.
    """
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    return OpenAI()


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


# ── Statistics ──────────────────────────────────────────────────────────

Z_95 = 1.959964  # standard normal quantile for a two-sided 95% interval


def wilson_interval(wins: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion, as percentages.

    Preferred over the normal approximation here: it stays inside [0, 1] and
    stays accurate for proportions near 0 or 1, where the naive interval
    breaks down.
    """
    if n == 0:
        return (0.0, 0.0)

    p = wins / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))

    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return (round(low * 100, 1), round(high * 100, 1))


def binomial_test_vs_chance(wins: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p=0.5.

    Under the null the distribution is symmetric, so the two-sided p-value is
    twice the tail beyond the observed count. Exact rather than approximate —
    n=198 is small enough to sum directly.
    """
    if n == 0:
        return 1.0

    k = max(wins, n - wins)  # fold to the upper tail
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def summarize_significance(wins_a: int, wins_b: int, label_a: str, label_b: str) -> dict:
    """Wilson CIs for both arms plus the significance verdict for the pair."""
    n = wins_a + wins_b
    p_value = binomial_test_vs_chance(wins_a, n)

    return {
        f"{label_a}_ci_95": wilson_interval(wins_a, n),
        f"{label_b}_ci_95": wilson_interval(wins_b, n),
        # 3 significant figures, so very small p-values survive instead of
        # rounding to 0.0
        "p_value": float(f"{p_value:.3g}"),
        "significant_at_05": bool(p_value < 0.05),
    }


def judge_pair(
    client: "OpenAI",
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
    client: "OpenAI",
) -> dict:
    """Run pairwise comparison across all prompts for one model pair."""
    from tqdm import tqdm

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
        **summarize_significance(wins_a, wins_b, label_a, label_b),
    }


def print_summary(all_comparisons: list[dict]):
    """Print win rates with confidence intervals and significance verdicts."""
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

        # 95% Wilson intervals — the honest width of each estimate
        a_lo, a_hi = comp.get(f"{a}_ci_95", (0, 0))
        b_lo, b_hi = comp.get(f"{b}_ci_95", (0, 0))
        print(f"  95% CI: {a:<6} [{a_lo:.1f}, {a_hi:.1f}]    {b:<6} [{b_lo:.1f}, {b_hi:.1f}]")

        p = comp.get("p_value", 1.0)
        verdict = "significant" if comp.get("significant_at_05") else "NOT significant"
        p_str = f"{p:.3g}" if p < 0.001 else f"{p:.4f}"
        print(f"  Exact binomial vs 50%: p = {p_str} -> {verdict} at alpha=0.05")

        if not comp.get("significant_at_05"):
            print(f"  ! Interval spans 50% — this gap is within noise at n={total}.")
        elif b_rate > a_rate:
            print(f"  → {b} wins ({b_rate - a_rate:.1f}pp improvement)")
        else:
            print(f"  → {a} wins ({a_rate - b_rate:.1f}pp)")

    print("\n" + "=" * 64)

    # Overall narrative
    if len(all_comparisons) >= 3:
        base_vs_dpo = all_comparisons[2]
        b = base_vs_dpo["model_b"]
        dpo_rate = base_vs_dpo[f"{b}_win_rate"]
        lo, hi = base_vs_dpo.get(f"{b}_ci_95", (0, 0))
        print(f"\n  End-to-end: DPO preferred over Base in {dpo_rate}% of comparisons "
              f"(95% CI [{lo:.1f}, {hi:.1f}])")
    elif all_comparisons:
        best = all_comparisons[0]
        a, b = best["model_a"], best["model_b"]
        b_rate = best[f"{b}_win_rate"]
        print(f"\n  {b} preferred over {a} in {b_rate}% of comparisons")
    print()


def recompute_stats(winrates_file: str = OUTPUT_FILE, output_file: str | None = None) -> list[dict]:
    """Re-derive CIs and p-values from an existing win-rates file.

    No API calls — the judgments are already counted, only the statistics are
    added. Use this to backfill stats onto results judged before this module
    computed them.
    """
    with open(winrates_file, "r", encoding="utf-8") as f:
        all_comparisons = json.load(f)

    for comp in all_comparisons:
        a, b = comp["model_a"], comp["model_b"]
        comp.update(summarize_significance(comp[f"{a}_wins"], comp[f"{b}_wins"], a, b))

    output_path = Path(output_file or winrates_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)
    print(f"Statistics recomputed and saved to {output_path}")

    print_summary(all_comparisons)
    return all_comparisons


def run_all_comparisons(
    results_file: str = RESULTS_FILE,
    output_file: str = OUTPUT_FILE,
) -> list[dict]:
    """Run all 3 pairwise comparisons and save results."""
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {results_file}")
    print(f"Judge model: {JUDGE_MODEL}\n")

    client = _load_api_client()
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

    print_summary(all_comparisons)
    return all_comparisons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise preference evaluation")
    parser.add_argument("--results", default=RESULTS_FILE, help="Path to results.json from benchmark.py")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output win rates path")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-derive CIs and p-values from an existing win-rates file (no API calls)",
    )
    args = parser.parse_args()

    if args.recompute:
        recompute_stats(winrates_file=args.output)
    else:
        run_all_comparisons(results_file=args.results, output_file=args.output)
