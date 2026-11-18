#!/usr/bin/env python3
"""
Command-line interface of the local RAG system (Tasks 1 & 2).

Examples
    python run_rag.py                                   # default question, CS handbook
    python run_rag.py -q "How many credits is the CS major?"
    python run_rag.py --chat                            # interactive chat with history
    python run_rag.py --data-dir data/pdf_demo --domain "the CS311 final project description" \
                      -q "How much is the Docker task worth?"
    python run_rag.py --rebuild --show-context          # re-index and print retrieved chunks
"""
import argparse
import sys

import config
from rag import RAGEngine, RAGSettings, ollama_models


def print_result(r, show_context=False):
    print("\nA: " + r["answer"] + "\n")
    if r["used_fallback"]:
        print(f"(no sufficiently relevant chunk found - best similarity {r['best_similarity']})")
    print("Sources:")
    for s in r["sources"]:
        page = f" p.{s['page']}" if s["page"] else ""
        print(f"   [{s['rank']}] {s['source']}{page}  similarity={s['score']}")
        if show_context:
            print("       " + s["content"][:400].replace("\n", "\n       ") + "\n")
    print(f"(retrieval {r['retrieve_s']} s, generation {r['generate_s']} s, model {r['model']})")


def main():
    p = argparse.ArgumentParser(description="Local RAG Q&A over PDF / TXT / MD documents")
    p.add_argument("-q", "--question", default=config.SAMPLE_QUESTION)
    p.add_argument("--data-dir", default=config.DATA_DIR, help="folder with .pdf/.txt/.md files")
    p.add_argument("--domain", default=config.DOMAIN, help="what the documents are about")
    p.add_argument("--model", default=config.LLM_MODEL, help="Ollama LLM, e.g. qwen2.5:3b")
    p.add_argument("--top-k", type=int, default=config.TOP_K)
    p.add_argument("--mode", default=config.RETRIEVAL_MODE, choices=["hybrid", "vector", "bm25"])
    p.add_argument("--rebuild", action="store_true", help="force re-indexing")
    p.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    p.add_argument("--chat", action="store_true", help="interactive multi-turn chat")
    a = p.parse_args()

    if not ollama_models(config.OLLAMA_URL):
        print(f"Cannot reach Ollama at {config.OLLAMA_URL}. Start it with 'ollama serve'.")
        sys.exit(1)

    settings = RAGSettings(data_dir=a.data_dir, domain=a.domain, llm_model=a.model,
                           top_k=a.top_k, retrieval_mode=a.mode)
    engine = RAGEngine(settings, rebuild=a.rebuild)
    print(f"Model: {a.model} | retrieval: {a.mode} | top-k: {a.top_k} | "
          f"chunks indexed: {engine.index_info.get('num_chunks')}")

    if not a.chat:
        print("Q: " + a.question)
        print_result(engine.ask(a.question), a.show_context)
        return

    history = []
    print("Interactive chat - type 'exit' to quit, 'reset' to clear history.")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "reset":
            history = []
            print("(history cleared)")
            continue
        if not q:
            continue
        r = engine.ask(q, history)
        print_result(r, a.show_context)
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": r["answer"]}]


if __name__ == "__main__":
    main()
