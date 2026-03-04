"""OpenAI API as preference judge for ranking response pairs."""

import json
import os
from itertools import combinations
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


load_dotenv()

JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator comparing two AI-generated responses to a question.

Evaluate both responses on these criteria:
1. **Helpfulness** — Does the response directly answer the question?
2. **Accuracy** — Is the information factually correct based on the context?
3. **Relevance** — Does the response stay focused on what was asked?
4. **Conciseness** — Is the response clear without unnecessary filler?

You MUST respond with valid JSON only, no other text. Use this exact format:
{
  "winner": "A" or "B",
  "reasoning": "Brief explanation of why the winner is better across the four criteria."
}"""

JUDGE_USER_TEMPLATE = """Context provided to both responses:
{context}

Question: {query}

--- Response A ---
{response_a}

--- Response B ---
{response_b}

Which response is better? Return JSON with "winner" ("A" or "B") and "reasoning"."""


def judge_pair(
    query: str,
    context: str,
    response_a: str,
    response_b: str,
    client: OpenAI,
    model: str = JUDGE_MODEL,
) -> dict:
    """Ask the judge model to pick the better response from a pair.

    Returns dict with keys: winner, chosen, rejected, reasoning.
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

    if winner == "A":
        chosen, rejected = response_a, response_b
    else:
        chosen, rejected = response_b, response_a

    return {
        "winner": winner,
        "chosen": chosen,
        "rejected": rejected,
        "reasoning": reasoning,
    }


def judge_candidates(collected_data: list[dict], model: str = JUDGE_MODEL) -> list[dict]:
    """Judge all pairwise combinations of candidates for each query.

    Takes the output of collect.collect_responses() and produces preference pairs.
    For N candidates per query, generates C(N,2) pairs.
    """
    client = OpenAI()
    preference_pairs = []

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
                preference_pairs.append({
                    "prompt": query,
                    "chosen": result["chosen"],
                    "rejected": result["rejected"],
                    "reasoning": result["reasoning"],
                })
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"  Skipping pair ({i},{j}) for query '{query[:50]}...': {e}")
                continue

    print(f"\nGenerated {len(preference_pairs)} preference pairs from {len(collected_data)} queries")
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
        print(f"  Reason:   {p['reasoning'][:120]}...")
