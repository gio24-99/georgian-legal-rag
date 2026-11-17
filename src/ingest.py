"""Read Georgian laws (.docx from matsne.gov.ge) and split them into articles (მუხლი).

    python src/ingest.py

Input:  data/docs/*.docx          (one file per law; the file name becomes the law's id)
Output: data/chunks.jsonl         (one line per article, or per part of a long article)

Each chunk keeps: law title, chapter (თავი), article number and title, so every answer
can later cite exactly where it came from.
"""
import argparse
import glob
import json
import os
import re
import statistics

from docx import Document

# "მუხლი 12. სათაური", "მუხლი 3¹. ...", "მუხლი 25." etc.
ARTICLE_RE = re.compile(r"^\s*მუხლი\s+(\d+[⁰¹²³⁴⁵⁶⁷⁸⁹]*)\s*\.?\s*(.*)$")
# "თავი I. ...", "თავი 2 ...", "თავი პირველი ..."
CHAPTER_RE = re.compile(r"^\s*თავი\s+\S+")

MAX_CHARS = 1500  # long articles are split into parts of about this size


def read_paragraphs(path: str) -> list[str]:
    doc = Document(path)
    paras = [p.text.strip() for p in doc.paragraphs]
    # some laws keep text inside tables
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paras.extend(p.text.strip() for p in cell.paragraphs)
    return [p for p in paras if p]


GENERIC_TITLE_RE = re.compile(r"^საქართველოს\s+(ორგანული\s+)?კანონი$")


def detect_title(paras: list[str]) -> str:
    """Matsne files often start with a generic line ("საქართველოს ორგანული კანონი") before the real name."""
    first = paras[0]
    if GENERIC_TITLE_RE.match(first) and len(paras) > 1:
        nxt = paras[1]
        # "საქართველოს კანონი" + "... შესახებ"  ->  "საქართველოს კანონი ... შესახებ"
        return f"{first} {nxt}" if nxt.endswith("შესახებ") else nxt
    if first.endswith("შესახებ") and not first.startswith("საქართველოს"):
        return f"კანონი {first}"
    return first


def split_long(text: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries so each part stays under max_chars."""
    parts, current = [], ""
    for para in text.split("\n"):
        if current and len(current) + len(para) + 1 > max_chars:
            parts.append(current)
            current = para
        else:
            current = f"{current}\n{para}" if current else para
    if current:
        parts.append(current)
    return parts


def parse_law(path: str) -> list[dict]:
    law_id = os.path.splitext(os.path.basename(path))[0]
    paras = read_paragraphs(path)
    law_title = detect_title(paras) if paras else law_id

    articles, chapter, current = [], "", None
    for p in paras:
        if CHAPTER_RE.match(p) and len(p) < 200:
            chapter = p
            continue
        m = ARTICLE_RE.match(p)
        if m:
            if current:
                articles.append(current)
            current = {"article": m.group(1), "title": m.group(2).strip(), "chapter": chapter, "lines": []}
        elif current:
            current["lines"].append(p)
    if current:
        articles.append(current)

    # The same article number can appear twice (quoted amendments, tables). Keep both, but with unique ids.
    seen: dict[str, int] = {}
    for a in articles:
        seen[a["article"]] = seen.get(a["article"], 0) + 1
        a["occurrence"] = seen[a["article"]]
    repeated = sorted({a["article"] for a in articles if seen[a["article"]] > 1}, key=lambda x: (len(x), x))
    if repeated:
        print(f"   WARNING {law_id}: article numbers appearing more than once: {', '.join(repeated)}")

    chunks = []
    for a in articles:
        body = "\n".join(a["lines"]).strip()
        header = f"{law_title}, მუხლი {a['article']}" + (f". {a['title']}" if a["title"] else "")
        parts = split_long(body, MAX_CHARS) if body else [""]
        for i, part in enumerate(parts, 1):
            chunks.append({
                "id": f"{law_id}-art{a['article']}"
                      + (f"-dup{a['occurrence']}" if a["occurrence"] > 1 else "")
                      + (f"-p{i}" if len(parts) > 1 else ""),
                "law_id": law_id,
                "law_title": law_title,
                "chapter": a["chapter"],
                "article": a["article"],
                "article_title": a["title"],
                # the header is part of the text, so retrieval also matches article titles
                "text": f"{header}\n{part}".strip(),
            })
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs_dir", default="data/docs")
    p.add_argument("--out", default="data/chunks.jsonl")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.docs_dir, "*.docx")))
    if not files:
        raise SystemExit(f"No .docx files in {args.docs_dir}. (Old .doc files: open in Word and 'Save As' .docx)")

    all_chunks = []
    for f in files:
        chunks = parse_law(f)
        all_chunks.extend(chunks)
        n_articles = len({c["article"] for c in chunks})
        lengths = [len(c["text"]) for c in chunks] or [0]
        print(f"\n== {os.path.basename(f)}")
        print(f"   title:    {chunks[0]['law_title'][:80] if chunks else '(no articles found!)'}")
        print(f"   articles: {n_articles}   chunks: {len(chunks)}   "
              f"chars avg/max: {int(statistics.mean(lengths))}/{max(lengths)}")
        if chunks:
            print(f"   first:    {chunks[0]['text'][:120]!r}")
            print(f"   last:     {chunks[-1]['text'][:120]!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(all_chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()