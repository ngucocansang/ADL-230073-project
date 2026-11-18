#!/usr/bin/env python3
"""
Evaluation of the CS-handbook chatbot (Task 2).

1) Retrieval ablation (fast, no LLM):
       python evaluate.py --retrieval
   -> eval/results/retrieval_ablation.md
   Hit@5 = a chunk from the correct handbook page is among the top-5 chunks.

2) End-to-end answer evaluation for several system configurations / LLMs:
       python evaluate.py --generate baseline improved-0.5b improved-1.5b improved-3b
   -> eval/results/answers_<config>.jsonl, eval/results/summary.md

Automatic labels (checked by hand afterwards, see report):
   correct     - every required fact appears in the answer
   incomplete  - some but not all required facts appear
   incorrect   - none of the required facts appear (or a hallucinated answer
                 to an out-of-scope question)
"""
import argparse
import json
import re
import time
from dataclasses import replace
from pathlib import Path

from rag import RAGEngine, RAGSettings, baseline_settings

HERE = Path(__file__).resolve().parent
QUESTIONS = json.loads((HERE / "eval" / "questions.json").read_text())
OUT = HERE / "eval" / "results"
OUT.mkdir(parents=True, exist_ok=True)

REFUSAL = re.compile(r"could not find|couldn't find|cannot find|can't find|not (mentioned|provided|"
                     r"specified|available|included|stated|listed)|no information|does not (contain|"
                     r"mention|provide|specify)|doesn't (contain|mention|provide)|not in the", re.I)


def norm(text: str) -> str:
    text = text.lower().replace("–", "-").replace("’", "'")
    return re.sub(r"\s+", " ", text)


def fact_present(answer: str, alternatives) -> bool:
    a = norm(answer)
    for alt in alternatives:
        alt = norm(alt)
        if re.search(r"(?<![a-z0-9])" + re.escape(alt) + r"(?:s|es)?(?![a-z0-9])", a):
            return True
    return False


def score_answer(q, answer: str):
    if q.get("out_of_scope"):
        ok = bool(REFUSAL.search(answer))
        return (1.0 if ok else 0.0), ("correct" if ok else "incorrect")
    facts = q["facts"]
    found = sum(fact_present(answer, f) for f in facts)
    frac = found / len(facts)
    label = "correct" if frac == 1 else ("incomplete" if frac > 0 else "incorrect")
    if frac < 1 and REFUSAL.search(answer):
        label = "refused" if frac == 0 else label
    return round(frac, 3), label


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
def improved(model: str, **kw) -> RAGSettings:
    return replace(RAGSettings(), llm_model=model, **kw)


GEN_CONFIGS = {
    "baseline": lambda: baseline_settings(),                       # sample code as given
    "baseline-1.5b": lambda: baseline_settings(llm_model="qwen2.5:1.5b"),
    "grounded-v1-0.5b": lambda: improved("qwen2.5:0.5b", prompt_style="grounded_v1"),
    "grounded-v1-1.5b": lambda: improved("qwen2.5:1.5b", prompt_style="grounded_v1"),
    "grounded-v1-3b": lambda: improved("qwen2.5:3b", prompt_style="grounded_v1"),
    "improved-0.5b": lambda: improved("qwen2.5:0.5b"),
    "improved-1.5b": lambda: improved("qwen2.5:1.5b"),
    "improved-3b": lambda: improved("qwen2.5:3b"),
    "improved-1.5b-c1000": lambda: improved("qwen2.5:1.5b", chunk_size=1000),
}

RETR_CONFIGS = [
    ("R0 baseline (1000/200, L2, no prefix, vector)", baseline_settings()),
    ("R1 + nomic task prefixes", baseline_settings(embed_prefixes=True)),
    ("R2 + cosine distance", baseline_settings(embed_prefixes=True, distance="cosine")),
    ("R3 + structure-aware chunking (1000/150)",
     baseline_settings(embed_prefixes=True, distance="cosine", structure_aware_split=True,
                       chunk_overlap=150)),
    ("R3b + follow-up query expansion (chat history)",
     baseline_settings(embed_prefixes=True, distance="cosine", structure_aware_split=True,
                       chunk_overlap=150, history_turns=2)),
    ("R3c + merge tiny chunks (<150 chars)",
     baseline_settings(embed_prefixes=True, distance="cosine", structure_aware_split=True,
                       chunk_overlap=150, history_turns=2, min_chunk_chars=150)),
    ("R4 + hybrid BM25+vector (chunk 1000)", replace(RAGSettings(), chunk_size=1000)),
    ("R5 final system: hybrid, chunk 800/150", RAGSettings()),
    ("R6 final, vector only", replace(RAGSettings(), retrieval_mode="vector")),
    ("R7 final, BM25 only", replace(RAGSettings(), retrieval_mode="bm25")),
    ("R8 final, chunk size 500", replace(RAGSettings(), chunk_size=500, chunk_overlap=100)),
    ("R9 final, chunk size 1500", replace(RAGSettings(), chunk_size=1500, chunk_overlap=200)),
]


def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t)


def run_retrieval():
    """Page Hit@k: a top-k chunk comes from a gold page (lenient).
    Evidence@k: a top-k chunk contains the exact evidence sentence (strict)."""
    rows = []
    qs = [q for q in QUESTIONS if q["pages"]]
    for name, s in RETR_CONFIGS:
        eng = RAGEngine(s, verbose=False)
        hit1 = hit5 = rr = ev5 = ev_rr = 0.0
        misses = []
        for q in qs:
            rq = eng.retrieval_query(q["question"], q.get("history"))
            sources, _ = eng.retrieve(rq, top_k=5)
            pages = [src["page"] for src in sources]
            rank = next((i + 1 for i, p in enumerate(pages) if p in q["pages"]), None)
            erank = next((i + 1 for i, src in enumerate(sources)
                          if q["evidence"] in _ws(src["content"])), None)
            hit1 += rank == 1
            hit5 += rank is not None
            rr += 1 / rank if rank else 0
            ev5 += erank is not None
            ev_rr += 1 / erank if erank else 0
            if not erank:
                misses.append(q["id"])
        n = len(qs)
        rows.append((name, eng.index_info.get("num_chunks"), hit1 / n, hit5 / n, rr / n,
                     ev5 / n, ev_rr / n, misses))
        print(f"{name:52s} chunks={rows[-1][1]:3d} page-hit@1={hit1/n:.2f} page-hit@5={hit5/n:.2f} "
              f"evidence@5={ev5/n:.2f} evidence-MRR={ev_rr/n:.2f} misses={misses}")
    md = ["# Retrieval ablation (30 answerable questions, top-5)\n",
          "Page-Hit = a retrieved chunk comes from the gold page (lenient). "
          "Evidence = a retrieved chunk contains the exact evidence sentence (strict).\n",
          "| Configuration | #chunks | Page-Hit@1 | Page-Hit@5 | Page-MRR | Evidence@5 | Evidence-MRR | Evidence missed |",
          "|---|---|---|---|---|---|---|---|"]
    for name, nc, h1, h5, mrr, e5, emrr, miss in rows:
        md.append(f"| {name} | {nc} | {h1:.2f} | {h5:.2f} | {mrr:.2f} | {e5:.2f} | {emrr:.2f} | "
                  f"{', '.join(miss) or '-'} |")
    (OUT / "retrieval_ablation.md").write_text("\n".join(md) + "\n")


def run_generation(names):
    for name in names:
        s = GEN_CONFIGS[name]()
        eng = RAGEngine(s, verbose=False)
        eng.ask("warm-up: what is this document about?")          # load model into memory
        path = OUT / f"answers_{name}.jsonl"
        with open(path, "w") as f:
            for q in QUESTIONS:
                t0 = time.time()
                r = eng.ask(q["question"], q.get("history"))
                frac, label = score_answer(q, r["answer"])
                rec = {"id": q["id"], "section": q["section"], "question": q["question"],
                       "reference": q["reference"], "answer": r["answer"],
                       "fact_recall": frac, "auto_label": label,
                       "used_fallback": r["used_fallback"], "best_similarity": r["best_similarity"],
                       "retrieved_pages": [x["page"] for x in r["sources"]],
                       "gold_pages": q["pages"], "seconds": round(time.time() - t0, 2),
                       "config": name, "model": s.llm_model}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{name}] {q['id']} {label:10s} {rec['seconds']:6.1f}s  {r['answer'][:90]!r}")
    summarize()


def rescore():
    """Re-compute automatic labels from the stored answers (no LLM calls)."""
    qmap = {q["id"]: q for q in QUESTIONS}
    for p in OUT.glob("answers_*.jsonl"):
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        for r in recs:
            r["fact_recall"], r["auto_label"] = score_answer(qmap[r["id"]], r["answer"])
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    summarize()


def load_manual():
    path = HERE / "eval" / "manual_review.json"
    return json.loads(path.read_text()) if path.exists() else {}


def final_label(rec, manual):
    """Manual label if the answer was re-labelled by hand, otherwise the automatic one."""
    o = manual.get(rec["config"], {}).get(rec["id"])
    if o:
        return o["label"]
    if rec["id"] == "Q24" and rec["auto_label"] != "refused":
        return "ambiguous"          # source text itself is wrong (see questions.json)
    return rec["auto_label"]


LABELS = ("correct", "incomplete", "incorrect", "ambiguous", "refused")


def summarize():
    manual = load_manual()
    files = sorted(OUT.glob("answers_*.jsonl"))
    order = list(GEN_CONFIGS)
    files.sort(key=lambda p: order.index(p.stem[8:]) if p.stem[8:] in order else 99)
    runs = []
    for p in files:
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if recs:
            runs.append(recs)
    md = ["# End-to-end evaluation summary (32 questions)\n",
          "Labels after manual review of every answer (`eval/manual_review.json`). "
          "`refused` = the system said it could not find the answer although the handbook contains it.\n",
          "| Config | LLM | Prompt | Correct | Incomplete | Incorrect | Ambiguous | Refused | "
          "Out-of-scope declined | Mean latency (s) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for recs in runs:
        labs = [final_label(r, manual) for r in recs]
        c = {k: labs.count(k) for k in LABELS}
        oos = sum(1 for r, l in zip(recs, labs) if r["section"] == "Out of scope" and l == "correct")
        prompt = ("baseline" if recs[0]["config"].startswith("baseline") else
                  "v1" if "v1" in recs[0]["config"] else "v2")
        md.append(f"| {recs[0]['config']} | {recs[0]['model']} | {prompt} | {c['correct']} | "
                  f"{c['incomplete']} | {c['incorrect']} | {c['ambiguous']} | {c['refused']} | "
                  f"{oos}/2 | {sum(r['seconds'] for r in recs)/len(recs):.1f} |")
    md += ["\n## Automatic (keyword) labels before manual review\n",
           "| Config | Correct | Incomplete | Incorrect | Refused | Mean fact recall |",
           "|---|---|---|---|---|---|"]
    for recs in runs:
        c = {k: sum(r["auto_label"] == k for r in recs) for k in LABELS}
        md.append(f"| {recs[0]['config']} | {c['correct']} | {c['incomplete']} | {c['incorrect']} | "
                  f"{c['refused']} | {sum(r['fact_recall'] for r in recs)/len(recs):.2f} |")
    cfgs = [r[0]["config"] for r in runs]
    per_q = {(r["config"], r["id"]): final_label(r, manual) for recs in runs for r in recs}
    md += ["\n## Per-question labels (manual)\n",
           "| ID | Question | " + " | ".join(cfgs) + " |", "|---|---|" + "---|" * len(cfgs)]
    for q in QUESTIONS:
        md.append(f"| {q['id']} | {q['question']} | " +
                  " | ".join(per_q.get((c, q["id"]), "") for c in cfgs) + " |")
    notes = [(cfg, qid, o) for cfg, d in manual.items() for qid, o in d.items()]
    if notes:
        md += ["\n## Manual review notes\n", "| Config | ID | Auto → manual | Note |", "|---|---|---|---|"]
        auto = {(r["config"], r["id"]): r["auto_label"] for recs in runs for r in recs}
        for cfg, qid, o in sorted(notes):
            md.append(f"| {cfg} | {qid} | {auto.get((cfg, qid), '?')} → {o['label']} | {o.get('note', '')} |")
    (OUT / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:4 + len(runs)]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generate", nargs="*", choices=list(GEN_CONFIGS))
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    if a.retrieval:
        run_retrieval()
    if a.generate:
        run_generation(a.generate)
    if a.rescore:
        rescore()
    elif a.summary:
        summarize()
