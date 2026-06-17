"""
UM-Dearborn RAG Evaluation using RAGAS
Measures: Faithfulness, Answer Relevancy, Context Precision, Context Recall

Run from project root:
    python3 scripts/evaluate.py

Optional flags:
    --questions   path to test questions JSON  (default: data/test_questions.json)
    --output      path to save report JSON     (default: data/eval_report.json)
    --sample      number of questions to run   (default: all)
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from rag_engine import VectorStore, RAGEngine

# ── RAGAS imports (v0.2) ───────────────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


def build_engine():
    chroma_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
    store = VectorStore(chroma_dir=chroma_path, collection_name="umdearborn")
    print(f"   Vector store loaded: {store.count()} chunks")
    engine = RAGEngine(vector_store=store)
    return engine


def run_rag(engine: RAGEngine, question: str, top_k: int = 5):
    """Run one question through the RAG and return answer + retrieved contexts."""
    answer, chunks = engine.chat(question, top_k=top_k)
    contexts = [c.text for c in chunks]
    engine.reset_memory()  # keep each eval question independent
    return answer, contexts


def build_ragas_llm_and_embeddings():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY not set in .env")

    # Use a separate, lighter model for RAGAS judging so its daily token quota
    # stays independent from the RAG generation model.  Fall back to the main
    # LLM_MODEL if the env var is not set.
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.1-8b-instant")
    print(f"   RAGAS judge LLM : {judge_model} (Groq)")
    print(f"   RAGAS embeddings: all-MiniLM-L6-v2 (local)")

    llm = LangchainLLMWrapper(ChatGroq(model=judge_model, api_key=groq_key))
    embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    return llm, embeddings


def main():
    parser = argparse.ArgumentParser(description="Evaluate UM-Dearborn RAG with RAGAS")
    parser.add_argument("--questions", default=os.path.join(os.path.dirname(__file__), '..', 'data', 'test_questions.json'))
    parser.add_argument("--output",    default=os.path.join(os.path.dirname(__file__), '..', 'data', 'eval_report.json'))
    parser.add_argument("--sample",    type=int,   default=None, help="Run only first N questions")
    parser.add_argument("--delay",     type=float, default=1.0,  help="Seconds between RAG calls (default: 1.0)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  UM-Dearborn RAG — RAGAS Evaluation")
    print("="*60)

    # ── Load test questions ────────────────────────────────────────
    with open(args.questions) as f:
        test_cases = json.load(f)
    if args.sample:
        test_cases = test_cases[:args.sample]
    print(f"\n[1/4] Loaded {len(test_cases)} test questions")

    # ── Build RAG engine ───────────────────────────────────────────
    print("\n[2/4] Loading RAG engine...")
    engine = build_engine()

    # ── Run all questions through RAG ──────────────────────────────
    print(f"\n[3/4] Running {len(test_cases)} questions through RAG...")
    samples = []
    for i, tc in enumerate(test_cases, 1):
        q = tc["question"]
        gt = tc["ground_truth"]
        print(f"  [{i:02d}/{len(test_cases)}] {q[:70]}...")
        try:
            answer, contexts = run_rag(engine, q)
            samples.append(SingleTurnSample(
                user_input=q,
                response=answer,
                retrieved_contexts=contexts,
                reference=gt,
            ))
        except Exception as e:
            print(f"         ⚠️  Skipped (error: {e})")
        time.sleep(args.delay)

    dataset = EvaluationDataset(samples=samples)
    print(f"\n   Collected {len(samples)} samples for evaluation")

    # ── Run RAGAS evaluation ───────────────────────────────────────
    print("\n[4/4] Running RAGAS scoring (this takes a few minutes)...")
    llm, embeddings = build_ragas_llm_and_embeddings()

    # Configure each metric with our Groq LLM + local embeddings
    faithfulness.llm             = llm
    answer_relevancy.llm         = llm
    answer_relevancy.embeddings  = embeddings
    context_precision.llm        = llm
    context_recall.llm           = llm

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(dataset=dataset, metrics=metrics)
    df = results.to_pandas()

    # ── Print report ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)

    score_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    labels = {
        "faithfulness":       "Faithfulness       (no hallucination)",
        "answer_relevancy":   "Answer Relevancy   (addresses question)",
        "context_precision":  "Context Precision  (chunks are relevant)",
        "context_recall":     "Context Recall     (found all needed info)",
    }

    import math
    overall = {}
    for col in score_cols:
        if col in df.columns:
            score = df[col].mean()
            if math.isnan(score):
                print(f"\n  ⏭️  {labels[col]}")
                print(f"      Score: N/A  (timed out — try again with --delay 3)")
                continue
            overall[col] = round(score, 4)
            filled = int(score * 20)
            bar = "█" * filled + "░" * (20 - filled)
            grade = "✅" if score >= 0.75 else "⚠️ " if score >= 0.5 else "❌"
            print(f"\n  {grade} {labels[col]}")
            print(f"      Score: {score:.4f}  [{bar}]  {score*100:.1f}%")

    avg = sum(overall.values()) / len(overall) if overall else 0
    print(f"\n  {'─'*50}")
    print(f"  Overall Average: {avg*100:.1f}%")
    print("="*60)

    # Per-question breakdown
    print("\n  PER-QUESTION BREAKDOWN:")
    print(f"  {'#':<4} {'Question':<50} {'Faith':>6} {'Relev':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'─'*4} {'─'*50} {'─'*6} {'─'*6} {'─'*6} {'─'*6}")
    def fmt(row, col):
        v = row.get(col)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "  N/A"
        return f"{float(v):.2f}"

    for i, row in df.iterrows():
        q = str(row.get("user_input", ""))[:48]
        print(f"  {i+1:<4} {q:<50} {fmt(row,'faithfulness'):>6} {fmt(row,'answer_relevancy'):>6} {fmt(row,'context_precision'):>6} {fmt(row,'context_recall'):>6}")

    # ── Save JSON report ───────────────────────────────────────────
    report = {
        "overall": overall,
        "average": round(avg, 4),
        "num_questions": len(samples),
        "per_question": df.to_dict(orient="records"),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Full report saved → {args.output}")
    print()


if __name__ == "__main__":
    main()
