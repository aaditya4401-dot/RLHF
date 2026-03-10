"""OpenAI API as preference judge for ranking response pairs.

Enforces strict quality-gap filtering: only pairs with a clear, confident
preference (confidence >= 4/5) are kept.  Ambiguous pairs are discarded
to prevent DPO mode collapse from noisy signal.
"""

import json
import os
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

# Minimum confidence (1-5) required to keep a pair.
MIN_CONFIDENCE = 4

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator comparing two AI-generated responses to an HR policy question.

Evaluate both responses on these criteria:
1. **Helpfulness** — Does the response directly answer the question?
2. **Accuracy** — Is the information factually correct based on the context?
3. **Relevance** — Does the response stay focused on what was asked?
4. **Conciseness** — Is the response clear without unnecessary filler?

You MUST respond with valid JSON only, no other text. Use this exact format:
{
  "winner": "A" or "B",
  "confidence": <integer 1-5>,
  "reasoning": "Brief explanation of why the winner is better across the four criteria."
}

Confidence scale:
- 5: One response is clearly superior across all criteria
- 4: One response is noticeably better on most criteria
- 3: One response is slightly better but it's a close call
- 2: Both responses are nearly equal in quality
- 1: Cannot meaningfully distinguish between the two"""

JUDGE_USER_TEMPLATE = """Context provided to both responses:
{context}

Question: {query}

--- Response A ---
{response_a}

--- Response B ---
{response_b}

Which response is better? Return JSON with "winner" ("A" or "B"), "confidence" (1-5), and "reasoning"."""


def judge_pair(
    query: str,
    context: str,
    response_a: str,
    response_b: str,
    client: OpenAI,
    model: str = JUDGE_MODEL,
) -> dict:
    """Ask the judge model to pick the better response from a pair.

    Returns dict with keys: winner, chosen, rejected, reasoning, confidence.
    """
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                context=context,
                query=query,
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

    # Parse the JSON verdict
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        if "```" in raw:
            json_str = raw.split("```")[1]
            if json_str.startswith("json"):
                json_str = json_str[4:]
            verdict = json.loads(json_str.strip())
        else:
            raise ValueError(f"Judge returned unparseable response: {raw}")

    winner = verdict["winner"].upper()
    reasoning = verdict.get("reasoning", "")
    confidence = int(verdict.get("confidence", 3))

    if winner == "A":
        chosen, rejected = response_a, response_b
    else:
        chosen, rejected = response_b, response_a

    return {
        "winner": winner,
        "chosen": chosen,
        "rejected": rejected,
        "reasoning": reasoning,
        "confidence": confidence,
    }


def judge_candidates(
    collected_data: list[dict],
    model: str = JUDGE_MODEL,
    min_confidence: int = MIN_CONFIDENCE,
) -> list[dict]:
    """Judge all pairwise combinations of candidates for each query.

    Takes the output of collect.collect_responses() and produces preference pairs.
    For N candidates per query, generates C(N,2) pairs.

    Only pairs where the judge's confidence >= min_confidence are kept.
    This ensures a clear quality gap between chosen and rejected responses,
    which is critical for stable DPO training.
    """
    client = OpenAI()
    preference_pairs = []
    skipped_low_confidence = 0
    skipped_identical = 0
    skipped_error = 0

    for item in tqdm(collected_data, desc="Judging queries"):
        query = item["query"]
        context = item["context"]
        candidates = item["candidates"]

        # Generate all unique pairs
        for i, j in combinations(range(len(candidates)), 2):
            resp_a = candidates[i]["response"]
            resp_b = candidates[j]["response"]

            # Skip if responses are nearly identical
            if resp_a.strip() == resp_b.strip():
                skipped_identical += 1
                continue

            try:
                result = judge_pair(
                    query=query,
                    context=context,
                    response_a=resp_a,
                    response_b=resp_b,
                    client=client,
                    model=model,
                )

                # Filter: only keep pairs with a clear quality gap
                if result["confidence"] < min_confidence:
                    skipped_low_confidence += 1
                    continue

                preference_pairs.append({
                    "prompt": query,
                    "chosen": result["chosen"],
                    "rejected": result["rejected"],
                    "reasoning": result["reasoning"],
                    "confidence": result["confidence"],
                })
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  Skipping pair ({i},{j}) for query '{query[:50]}...': {e}")
                skipped_error += 1
                continue

    total_evaluated = len(preference_pairs) + skipped_low_confidence + skipped_identical + skipped_error
    print(f"\n--- Judging Summary ---")
    print(f"Total pairs evaluated:    {total_evaluated}")
    print(f"Kept (confidence >= {min_confidence}):  {len(preference_pairs)}")
    print(f"Skipped (low confidence): {skipped_low_confidence}")
    print(f"Skipped (identical):      {skipped_identical}")
    print(f"Skipped (parse errors):   {skipped_error}")
    return preference_pairs


def save_judgments(pairs: list[dict], output_path: str | Path | None = None):
    """Save judged preference pairs to JSON."""
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "data" / "preference_data" / "judgments.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(pairs)} preference pairs to {output_path}")
    return output_path


if __name__ == "__main__":
    # Load candidates from collect.py output
    candidates_path = Path(__file__).resolve().parents[2] / "data" / "preference_data" / "candidates.json"
    if not candidates_path.exists():
        print(f"No candidates found at {candidates_path}")
        print("Run `python -m src.preference.collect` first.")
        raise SystemExit(1)

    with open(candidates_path, "r", encoding="utf-8") as f:
        collected_data = json.load(f)

    print(f"Loaded {len(collected_data)} queries with candidates")
    pairs = judge_candidates(collected_data)
    save_judgments(pairs)

    # Print a few examples
    for p in pairs[:3]:
        print(f"\nPrompt: {p['prompt'][:80]}...")
        print(f"  Chosen:   {p['chosen'][:80]}...")
        print(f"  Rejected: {p['rejected'][:80]}...")
        print(f"  Confidence: {p['confidence']}/5")
        print(f"  Reason:   {p['reasoning'][:120]}...")
