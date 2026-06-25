"""
Deliberate experimentation harness for DocQA.QueryService.

Run from DocQA.QueryService/ with your .env in place:
    python experiments.py [--top-k] [--temperature] [--scores] [--all]

Each experiment prints what it found and WHY that result makes sense,
so you walk away understanding the system, not just seeing numbers.
"""

import argparse
import os
import sys
import textwrap
import time

# ── load .env before importing app code ──────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from app.services.retrieval import embed_query, retrieve_chunks
from app.services.generation import build_context_block
from app.core.azure_clients import get_openai_client
from app.core.config import get_settings

DIVIDER = "=" * 70
THIN    = "-" * 70

# ── helpers ──────────────────────────────────────────────────────────────────

def _generate(question: str, chunks, temperature: float) -> str:
    settings = get_settings()
    client = get_openai_client()
    context_block = build_context_block(chunks)
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using the context above, citing chunk numbers like [1], [2] where relevant."
    )
    resp = client.chat.completions.create(
        model=settings.azure_openai_chat_deployment,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a documentation assistant for Microsoft Semantic Kernel. "
                    "Answer the user's question using ONLY the provided context chunks. "
                    "If the context does not contain enough information to answer, say so "
                    "clearly rather than guessing."
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content or ""


def _print_score_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.4f}"


# ── experiment 1: retrieval score sweep ──────────────────────────────────────

def experiment_scores():
    """
    Show the actual cosine-similarity scores for several questions.
    This teaches you what "relevant" looks like numerically: scores near 1.0
    mean the query and chunk vectors point in almost the same direction in
    embedding space; scores near 0.0 mean they are unrelated.
    """
    questions = [
        ("Strong match",    "What is a plugin in Semantic Kernel?"),
        ("Weaker match",    "How does automatic function calling work?"),
        ("Out-of-domain",   "What is the airspeed velocity of an unladen swallow?"),
        ("Borderline",      "How do I handle errors in a chat loop?"),
    ]

    print(f"\n{DIVIDER}")
    print("EXPERIMENT 1 — Retrieval score inspection (what cosine similarity looks like)")
    print(DIVIDER)
    print(
        "Each bar shows the similarity score for a retrieved chunk.\n"
        "Azure AI Search normalises HNSW cosine scores to [0, 1].\n"
        "1.0 = identical direction; 0.0 = orthogonal (unrelated).\n"
    )

    for label, q in questions:
        chunks = retrieve_chunks(q, top_k=5)
        print(f"\n[{label}] Q: {q}")
        print(THIN)
        for i, c in enumerate(chunks, 1):
            source = (c.source or c.id)[:50]
            print(f"  #{i} {_print_score_bar(c.score)}  {source}")
        if chunks:
            gap = chunks[0].score - chunks[-1].score
            print(f"     Score spread (top vs #5): {gap:.4f}")
        print()


# ── experiment 2: top_k sweep ─────────────────────────────────────────────────

def experiment_top_k():
    """
    Retrieve the same question with top_k = 3, 5, 10, 20.
    Watch how: (a) scores at the tail drop off, (b) the context window grows.
    More chunks = more context for the LLM, but also more noise and cost.
    The sweet spot depends on how focused your corpus is.
    """
    question = "How do I add an AI service to the kernel?"
    top_k_values = [3, 5, 10, 20]

    print(f"\n{DIVIDER}")
    print("EXPERIMENT 2 — top_k sweep  (more context vs. noise tradeoff)")
    print(DIVIDER)
    print(f"Question: {question}\n")

    for k in top_k_values:
        chunks = retrieve_chunks(question, top_k=k)
        scores = [c.score for c in chunks]
        total_chars = sum(len(c.content) for c in chunks)
        print(f"  top_k={k:>2}  │  scores: {scores[0]:.3f} … {scores[-1]:.3f}"
              f"  │  total context chars: {total_chars:,}")

    print(
        "\nWhat to notice:\n"
        "  - The first few chunks are always the same; marginal chunks get noisier.\n"
        "  - At top_k=20 the context is many thousands of characters — is that all relevant?\n"
        "  - Context window cost grows linearly; retrieval noise grows super-linearly.\n"
    )


# ── experiment 3: temperature variance ───────────────────────────────────────

def experiment_temperature():
    """
    Run the same question at temperature=0.0 and temperature=0.7, three times each.
    temperature=0.0 → greedy decoding → deterministic / consistent.
    temperature=0.7 → sampled → creative but variable.
    For a grounded Q&A system, consistency usually beats creativity.
    """
    question = "What is the purpose of the kernel in Semantic Kernel?"
    temps = [0.0, 0.7]
    runs = 2

    print(f"\n{DIVIDER}")
    print("EXPERIMENT 3 — Temperature variance  (determinism vs. creativity)")
    print(DIVIDER)
    print(f"Question: {question}")
    print("Running each temperature twice to show variance...\n")

    chunks = retrieve_chunks(question, top_k=5)

    for temp in temps:
        print(f"\n── temperature={temp} ──────────────────────────────────────")
        answers = []
        for run in range(1, runs + 1):
            ans = _generate(question, chunks, temperature=temp)
            answers.append(ans)
            print(f"\n  Run {run}:")
            for line in textwrap.wrap(ans, width=66):
                print(f"    {line}")
            time.sleep(0.5)

        # crude similarity check: word overlap between runs
        if len(answers) == 2:
            w1 = set(answers[0].lower().split())
            w2 = set(answers[1].lower().split())
            overlap = len(w1 & w2) / max(len(w1 | w2), 1)
            print(f"\n  Word overlap between runs: {overlap:.0%}")

    print(
        "\nWhat to notice:\n"
        "  - At 0.0 the two answers should be nearly identical.\n"
        "  - At 0.7 they may start with different phrases or emphasise different sources.\n"
        "  - For a grounded RAG system, that variance is a liability, not a feature.\n"
    )


# ── experiment 4: unanswerable questions ─────────────────────────────────────

def experiment_unanswerable():
    """
    Ask things the corpus can't answer, then inspect retrieval scores.
    Key question: does the system confidently hallucinate, or does it refuse?
    Also look at the scores — out-of-domain queries still surface chunks,
    but with lower scores. At what threshold does a retrieved chunk become noise?
    """
    questions = [
        "What is the airspeed velocity of an unladen swallow?",
        "How much does Azure OpenAI cost per month?",
        "How do I deploy a Semantic Kernel app to Kubernetes?",
    ]

    print(f"\n{DIVIDER}")
    print("EXPERIMENT 4 — Unanswerable / out-of-domain questions")
    print(DIVIDER)
    print("Watching for: hallucination vs. grounded refusal, and score levels.\n")

    chunks_buf = {}
    for q in questions:
        chunks = retrieve_chunks(q, top_k=3)
        chunks_buf[q] = chunks
        top_score = chunks[0].score if chunks else 0
        print(f"Q: {q}")
        print(f"   Top retrieved score: {top_score:.4f}")
        print(f"   Top source: {(chunks[0].source or chunks[0].id)[:60] if chunks else 'none'}")

    print("\nNow generating answers — watch whether grounding holds...\n")
    for q, chunks in chunks_buf.items():
        answer = _generate(q, chunks, temperature=0.0)
        print(f"\nQ: {q}")
        print(f"A: {answer[:400]}{'...' if len(answer) > 400 else ''}")

    print(
        "\nWhat to notice:\n"
        "  - Even off-topic queries retrieve something — HNSW always returns k results.\n"
        "  - Low scores (< ~0.75) signal the retrieved chunks are probably noise.\n"
        "  - The system prompt says to refuse if context is insufficient — does it?\n"
        "  - If it hallucinates despite the grounding instruction, that is a prompt failure.\n"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DocQA experiment harness")
    parser.add_argument("--scores",      action="store_true", help="Experiment 1: score inspection")
    parser.add_argument("--top-k",       action="store_true", help="Experiment 2: top_k sweep")
    parser.add_argument("--temperature", action="store_true", help="Experiment 3: temperature variance")
    parser.add_argument("--unanswerable",action="store_true", help="Experiment 4: unanswerable queries")
    parser.add_argument("--all",         action="store_true", help="Run all experiments")
    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        print(
            "\nQuick start:\n"
            "  python experiments.py --scores         # see cosine similarity numbers\n"
            "  python experiments.py --top-k          # see context size tradeoff\n"
            "  python experiments.py --temperature    # see determinism vs. creativity\n"
            "  python experiments.py --unanswerable   # see grounding under pressure\n"
            "  python experiments.py --all            # run everything\n"
        )
        sys.exit(0)

    if args.all or args.scores:
        experiment_scores()
    if args.all or args.top_k:
        experiment_top_k()
    if args.all or args.temperature:
        experiment_temperature()
    if args.all or args.unanswerable:
        experiment_unanswerable()


if __name__ == "__main__":
    main()
