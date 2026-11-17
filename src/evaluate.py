"""Evaluate the assistant on eval/questions.csv: RAG vs. no RAG, for one or more LLMs.

    python src/evaluate.py --limit 3                 # quick test on 3 questions
    python src/evaluate.py                           # full run, gemma3:12b
    python src/evaluate.py --llms gemma3:12b gemma3:4b qwen3:8b

Measures, per configuration:
  - answer accuracy      (LLM judge compares the answer with the answer key)
  - retrieval hit rate   (was the expected article among the articles given to the LLM?)
  - citation validity    (no citations to sources that don't exist)
  - correct-source rate  (did the answer cite the expected article?)
  - average seconds per answer
Results: results/eval_<time>.csv (every answer) and results/summary_<time>.md (tables).
"""
import argparse
import csv
import os
import time

import ollama

from rag import answer

JUDGE_PROMPT = """You are grading an answer to a question about Georgian law. All texts are in Georgian.

Question: {question}
Reference answer (correct): {key}
Candidate answer: {answer}

Rules:
- If the reference answer is "not_in_docs": the candidate is CORRECT only if it says it could not find the answer
  or does not know. It is INCORRECT if it states a specific answer.
- Otherwise the candidate is CORRECT if it contains the key facts of the reference answer (numbers, who, what,
  yes/no) and does not contradict it. Extra correct details are fine. Missing a key fact or stating a wrong number
  makes it INCORRECT.
- Units and conditions matter: "working days" (სამუშაო დღე) and "calendar days" (კალენდარული დღე) are different;
  a wrong unit or a missing essential condition makes the answer INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def judge(question: str, key: str, ans: str, judge_model: str) -> bool:
    resp = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(question=question, key=key, answer=ans)}],
        options={"temperature": 0},
    )
    verdict = resp["message"]["content"].strip().upper()
    return "INCORRECT" not in verdict and "CORRECT" in verdict


def with_retry(fn, tries: int = 4, wait: int = 15):
    """Ollama can crash when it swaps models on a small GPU; wait and try again instead of stopping the run."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except ollama.ResponseError as e:
            if attempt == tries:
                raise
            print(f"   Ollama error ({str(e)[:80]}...), retry {attempt}/{tries - 1} in {wait}s")
            time.sleep(wait)


def pct(xs: list) -> str:
    xs = [x for x in xs if x is not None]
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "–"


def write_summary(rows: list[dict], llms: list[str], stamp: str, header: str):
    modes = [m for m in ("rag", "no_rag") if any(r["mode"] == m for r in rows)]
    lines = ["| LLM | Mode | Answer accuracy | Retrieval hit@k | Valid citations | Cites expected article | Avg s |",
             "|---|---|---|---|---|---|---|"]
    for llm in llms:
        for mode in modes:
            rs = [r for r in rows if r["llm"] == llm and r["mode"] == mode]
            if not rs:
                continue
            avg = sum(r["seconds"] for r in rs) / len(rs)
            lines.append(f"| {llm} | {mode} | {pct([r['correct'] for r in rs])} ({len(rs)} q) | "
                         f"{pct([r['retrieval_hit'] for r in rs])} | {pct([r['valid_citations'] for r in rs])} | "
                         f"{pct([r['cited_expected'] for r in rs])} | {avg:.1f} |")

    types = sorted({r["type"] for r in rows})
    lines += ["", "Answer accuracy by question type:", "",
              "| LLM | Mode | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types)]
    for llm in llms:
        for mode in modes:
            if not any(r["llm"] == llm and r["mode"] == mode for r in rows):
                continue
            cells = [pct([r["correct"] for r in rows if r["llm"] == llm and r["mode"] == mode and r["type"] == t])
                     for t in types]
            lines.append(f"| {llm} | {mode} | " + " | ".join(cells) + " |")

    summary = "\n".join(lines)
    path = f"results/summary_{stamp}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{header}\n\n{summary}\n")
    print("\n" + summary)
    print(f"\nSaved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", default="eval/questions.csv")
    p.add_argument("--llms", nargs="+", default=["gemma3:12b"])
    p.add_argument("--judge", default="gemma3:12b")
    p.add_argument("--embed_model", default="bge-m3")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (quick test)")
    p.add_argument("--summarize", nargs="+", metavar="CSV",
                   help="don't run anything: build the summary from existing results/eval_*.csv files "
                        "(later files win for the same llm/mode/question)")
    args = p.parse_args()

    if args.summarize:
        merged = {}
        for path in args.summarize:
            with open(path, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    for k in ("correct", "retrieval_hit", "valid_citations", "cited_expected"):
                        r[k] = {"True": True, "False": False}.get(r[k])
                    r["seconds"] = float(r["seconds"])
                    merged[(r["llm"], r["mode"], r["id"])] = r
        rows = list(merged.values())
        llms = list(dict.fromkeys(r["llm"] for r in rows))
        write_summary(rows, llms, time.strftime("%Y%m%d-%H%M"),
                      f"merged from {len(args.summarize)} file(s), judge={args.judge}")
        return

    with open(args.questions, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))[: args.limit]

    stamp = time.strftime("%Y%m%d-%H%M")
    os.makedirs("results", exist_ok=True)
    rows_path = f"results/eval_{stamp}.csv"
    fields = ["id", "type", "llm", "mode", "correct", "retrieval_hit", "valid_citations", "cited_expected",
              "seconds", "question", "answer_key", "answer", "sources"]

    rows = []
    with open(rows_path, "w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for llm in args.llms:
            for mode in ("rag", "no_rag"):
                for i, q in enumerate(questions, 1):
                    r = with_retry(lambda: answer(q["question"], llm, args.embed_model, args.k,
                                                  use_rag=(mode == "rag")))
                    expected = set(filter(None, q["expected_articles"].split(";")))
                    in_docs = q["type"] != "not_in_docs"

                    def is_expected(src):
                        return src["law_id"] == q["expected_law"] and src["article"] in expected

                    retrieval_hit = any(is_expected(s) for s in r["sources"]) if (mode == "rag" and in_docs) else None
                    cited_expected = (any(is_expected(r["sources"][n - 1]) for n in r["cited"])
                                      if (mode == "rag" and in_docs) else None)
                    row = {
                        "id": q["id"], "type": q["type"], "llm": llm, "mode": mode,
                        "correct": with_retry(lambda: judge(q["question"], q["answer_key"], r["answer"], args.judge)),
                        "retrieval_hit": retrieval_hit,
                        "valid_citations": (not r["invalid_citations"]) if mode == "rag" else None,
                        "cited_expected": cited_expected,
                        "seconds": r["seconds"],
                        "question": q["question"], "answer_key": q["answer_key"], "answer": r["answer"],
                        "sources": "; ".join(f"{s['law_id']} {s['article']}" for s in r["sources"]),
                    }
                    rows.append(row)
                    writer.writerow(row)
                    out.flush()
                    print(f"[{llm} {mode}] {i}/{len(questions)} {q['id']}: "
                          f"{'OK ' if row['correct'] else 'BAD'}  ({r['seconds']}s)")

    write_summary(rows, args.llms, stamp,
                  f"{len(questions)} questions, k={args.k}, embeddings={args.embed_model}, judge={args.judge}")
    print(f"Saved {rows_path}")

if __name__ == "__main__":
    main()