"""Generate multiple candidate responses per query using OpenAI with varied settings."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.rag.retriever import load_vectorstore, retrieve
from src.rag.generate import format_context


load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Sample queries for testing the pipeline end-to-end ---

SAMPLE_QUERIES = [
    # --- Original 20 ---
    "What is the company's policy on remote work?",
    "How do I apply for parental leave?",
    "What expenses are reimbursable during business travel?",
    "What is the process for filing a workplace harassment complaint?",
    "How many sick days am I entitled to per year?",
    "What are the steps in the employee onboarding process?",
    "How does the annual performance review process work?",
    "What is the policy on PTO carryover to the next year?",
    "How do I submit my resignation and what is the notice period?",
    "What health insurance benefits are available to employees?",
    "What is the company's code of conduct regarding conflicts of interest?",
    "How do I request a workplace accommodation under ADA?",
    "What training and professional development programs are offered?",
    "What is the policy on overtime pay and compensatory time off?",
    "How does the company handle confidential information and NDAs?",
    "What are the eligibility requirements for the 401(k) retirement plan?",
    "What is the dress code policy for the office?",
    "How do I report a safety hazard in the workplace?",
    "What is the process for an internal job transfer or promotion?",
    "What are the guidelines for using company equipment for personal use?",
    # --- Added queries: Benefits & Compensation ---
    "How does the employee stock option or equity program work?",
    "What dental and vision insurance plans are offered?",
    "How do I enroll in or change my benefits during open enrollment?",
    "What is the company match percentage for the retirement plan?",
    "Are there any employee discount programs or perks?",
    "How does the tuition reimbursement program work?",
    "What life insurance and disability coverage does the company provide?",
    "How is the annual bonus or incentive compensation calculated?",
    # --- Added queries: Leave & Time Off ---
    "What is the bereavement leave policy?",
    "How do I request a leave of absence for medical reasons (FMLA)?",
    "What is the policy on jury duty leave?",
    "Can I take unpaid leave and what is the approval process?",
    "How does the sabbatical or extended leave program work?",
    "What is the policy on mental health days or wellness leave?",
    "How do I request time off for volunteering or community service?",
    # --- Added queries: Workplace Policies ---
    "What is the company's anti-discrimination and equal opportunity policy?",
    "How does the grievance or dispute resolution process work?",
    "What is the policy on moonlighting or outside employment?",
    "What are the rules around workplace relationships and dating policies?",
    "How does the company handle substance abuse or drug testing?",
    "What is the social media policy for employees?",
    "What are the data privacy and personal information handling policies?",
    "What is the policy on working from a different state or country?",
    # --- Added queries: Career & Development ---
    "How does the mentorship or buddy program work for new hires?",
    "What is the process for requesting a salary review or raise?",
    "How are job levels and career ladders structured?",
    "What leadership development or management training is available?",
    "How do I apply for an international assignment or relocation?",
    # --- Added queries: Operations & Compliance ---
    "What is the policy on visitor access and guest badges?",
    "How do I request ergonomic equipment or a standing desk?",
    "What is the emergency evacuation procedure for the office?",
    "How does the company handle whistleblower protections?",
    "What are the rules for expensing home office equipment?",
    "What is the company's policy on background checks for new hires?",
    "How do I access the employee assistance program (EAP)?",
]

# --- System prompt variations for diversity ---

SYSTEM_PROMPTS = [
    (
        "You are an HR policy expert. Answer the question using only "
        "the provided context. Be detailed and precise."
    ),
    (
        "You are a helpful HR assistant. Based on the context below, give a "
        "clear and concise answer. Use bullet points where appropriate."
    ),
    (
        "You are a knowledgeable human resources professional. Answer the question from "
        "the context provided. Keep your response brief and to the point."
    ),
]

TEMPERATURES = [0.3, 0.7, 1.0]


def generate_candidates(
    query: str,
    context: str,
    client: OpenAI,
    model: str = MODEL,
    system_prompts: list[str] = SYSTEM_PROMPTS,
    temperatures: list[float] = TEMPERATURES,
) -> list[dict]:
    """Generate multiple response candidates for a single query.

    Produces one response per (system_prompt, temperature) combination.
    """
    candidates = []

    for temp in temperatures:
        for sp in system_prompts:
            messages = [
                {"role": "system", "content": sp},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
            )
            answer = response.choices[0].message.content
            candidates.append({
                "response": answer,
                "temperature": temp,
                "system_prompt": sp,
                "model": model,
            })

    return candidates


def collect_responses(
    queries: list[str],
    vectorstore=None,
    model: str = MODEL,
    k: int = 4,
) -> list[dict]:
    """Run the full collection pipeline: retrieve context, generate candidates.

    Returns a list of dicts, each containing the query, context, and candidate responses.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    client = OpenAI()
    results = []

    for query in tqdm(queries, desc="Collecting responses"):
        docs = retrieve(query, vectorstore=vectorstore, k=k)
        context = format_context(docs)

        candidates = generate_candidates(
            query=query,
            context=context,
            client=client,
            model=model,
        )

        results.append({
            "query": query,
            "context": context,
            "candidates": candidates,
        })

    return results


def save_candidates(results: list[dict], output_path: str | Path | None = None):
    """Save collected candidate responses to JSON."""
    if output_path is None:
        output_path = Path(__file__).resolve().parents[2] / "data" / "preference_data" / "candidates.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip non-serializable context_docs before saving
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} query results to {output_path}")
    return output_path


if __name__ == "__main__":
    print(f"Running collection for {len(SAMPLE_QUERIES)} queries...")
    print(f"Generating {len(TEMPERATURES) * len(SYSTEM_PROMPTS)} candidates per query\n")

    results = collect_responses(SAMPLE_QUERIES)
    path = save_candidates(results)

    # Print summary
    for r in results[:2]:
        print(f"\nQuery: {r['query']}")
        print(f"  Candidates generated: {len(r['candidates'])}")
        for i, c in enumerate(r["candidates"]):
            print(f"  [{i}] temp={c['temperature']}, response={c['response'][:80]}...")
