"""Answer a question about Georgian law: retrieve articles, then let the LLM answer only from them.

    python src/rag.py "რამდენი დღე ანაზღაურებადი შვებულება ეკუთვნის დასაქმებულს?"
    python src/rag.py "..." --no_rag            # same model without documents (for comparison)
    python src/rag.py "..." --llm gemma3:4b     # try another model
"""
import argparse
import json
import re
import time
from functools import lru_cache

import ollama

from search import search

CHUNKS_PATH = "data/chunks.jsonl"
MAX_CONTEXT_CHARS = 14000  # keeps the prompt inside the model's context window


def base_id(chunk_id: str) -> str:
    """'law-art13-p2' -> 'law-art13' (all parts of one article share a base id)."""
    return re.sub(r"-p\d+$", "", chunk_id)


@lru_cache(maxsize=1)
def load_articles() -> dict:
    """Full text of every article, rebuilt from its parts in chunks.jsonl."""
    articles = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            b = base_id(c["id"])
            if b not in articles:
                articles[b] = dict(c, id=b)
            else:  # later parts: drop the repeated header line
                articles[b]["text"] += "\n" + c["text"].split("\n", 1)[-1]
    return articles


def retrieve_articles(question: str, embed_model: str, k_chunks: int, max_articles: int) -> list[dict]:
    """Parent-document retrieval: search small chunks, return the whole articles they belong to."""
    articles, out, total = load_articles(), [], 0
    for hit in search(question, embed_model, k_chunks):
        b = base_id(hit["id"])
        if any(a["id"] == b for a in out):
            continue
        art = dict(articles[b], score=hit["score"])
        if out and total + len(art["text"]) > MAX_CONTEXT_CHARS:
            continue
        out.append(art)
        total += len(art["text"])
        if len(out) == max_articles:
            break
    return out

SYSTEM_RAG = """შენ ხარ ასისტენტი, რომელიც პასუხობს კითხვებს საქართველოს კანონმდებლობის შესახებ.
წესები:
1. უპასუხე მხოლოდ ქვემოთ მოცემული კანონის მუხლების საფუძველზე. ნუ გამოიყენებ საკუთარ ცოდნას.
2. ყოველ მტკიცებას მიუთითე წყაროს ნომერი კვადრატულ ფრჩხილებში, მაგალითად: [1] ან [2][3]. გამოიყენე მხოლოდ ქვემოთ მოცემული წყაროების ნომრები. თავად მუხლის ნომერი ციტირებაში არ დაწერო.
3. თუ მოცემულ მუხლებში პასუხი არ არის, თქვი: „მოცემულ კანონებში ამ კითხვაზე პასუხი ვერ ვიპოვე.“
4. ჯერ ჩამოაყალიბე ზოგადი წესი, შემდეგ — პირობები და გამონაკლისები. თუ წესი მოქმედებს მხოლოდ გარკვეულ შემთხვევებში, ეს აუცილებლად აღნიშნე.
5. უპასუხე ქართულად, მოკლედ და ზუსტად."""

SYSTEM_NO_RAG = "შენ ხარ ასისტენტი, რომელიც პასუხობს კითხვებს საქართველოს კანონმდებლობის შესახებ. უპასუხე ქართულად, მოკლედ და ზუსტად."


def answer(question: str, llm: str = "gemma3:12b", embed_model: str = "bge-m3", k: int = 4,
           use_rag: bool = True, k_chunks: int = 10) -> dict:
    """k = how many whole articles go to the LLM; k_chunks = how many chunks the search looks at."""
    start = time.time()
    hits = retrieve_articles(question, embed_model, k_chunks, k) if use_rag else []

    if use_rag:
        context = "\n\n".join(f"[{i}] {h['text']}" for i, h in enumerate(hits, 1))
        messages = [
            {"role": "system", "content": SYSTEM_RAG},
            {"role": "user", "content": f"კანონის მუხლები:\n\n{context}\n\nკითხვა: {question}"},
        ]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_NO_RAG},
            {"role": "user", "content": question},
        ]

    # temperature 0 -> same answer every run (needed for a fair evaluation);
    # bigger context window because Georgian text uses many tokens;
    # num_predict caps the answer length so a model can't loop forever
    options = {"temperature": 0, "num_ctx": 12288, "num_predict": 1024}
    try:
        # thinking models (qwen3): answer directly, without the long reasoning phase
        resp = ollama.chat(model=llm, messages=messages, options=options, think=False)
    except (TypeError, ollama.ResponseError):
        # older ollama library, or a model that doesn't support the think option
        resp = ollama.chat(model=llm, messages=messages, options=options)
    # "thinking" models (e.g. qwen3) may put their reasoning in <think>...</think>; keep only the answer
    text = re.sub(r"<think>.*?</think>", "", resp["message"]["content"], flags=re.S).strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return {
        "question": question,
        "answer": text,
        # citation check: which sources the answer used, and any numbers that don't exist
        "cited": [n for n in cited if 1 <= n <= len(hits)],
        "invalid_citations": [n for n in cited if not 1 <= n <= len(hits)],
        "sources": hits,
        "llm": llm,
        "use_rag": use_rag,
        "seconds": round(time.time() - start, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("question")
    p.add_argument("--llm", default="gemma3:12b")
    p.add_argument("--embed_model", default="bge-m3")
    p.add_argument("-k", type=int, default=4, help="number of whole articles given to the LLM")
    p.add_argument("--no_rag", action="store_true")
    args = p.parse_args()

    r = answer(args.question, args.llm, args.embed_model, args.k, use_rag=not args.no_rag)
    print(f"\n{r['answer']}\n")
    if r["sources"]:
        print("წყაროები / Sources:")
        for i, h in enumerate(r["sources"], 1):
            mark = "*" if i in r["cited"] else " "
            print(f" {mark}[{i}] {h['law_title']}, მუხლი {h['article']}. {h['article_title']}  (score {h['score']:.2f})")
        print("  (* = cited in the answer)")
        if r["invalid_citations"]:
            print(f"  WARNING: answer cites sources that don't exist: {r['invalid_citations']}")
    print(f"\n({r['llm']}, {'RAG' if r['use_rag'] else 'no RAG'}, {r['seconds']}s)")


if __name__ == "__main__":
    main()