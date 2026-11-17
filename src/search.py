"""Search the law index (retrieval only, no LLM yet) to check that the right articles come back.

    python src/search.py "რამდენი დღე შვებულება ეკუთვნის დასაქმებულს?"
"""
import argparse

import chromadb

from index import CHROMA_DIR, collection_name, embed


def search(query: str, embed_model: str = "bge-m3", k: int = 5):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(collection_name(embed_model))
    res = col.query(query_embeddings=embed([query], embed_model), n_results=k,
                    include=["documents", "metadatas", "distances"])
    return [
        {"id": cid, "text": doc, "score": 1 - dist, **meta}
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--embed_model", default="bge-m3")
    p.add_argument("-k", type=int, default=5)
    args = p.parse_args()

    for i, hit in enumerate(search(args.query, args.embed_model, args.k), 1):
        print(f"\n[{i}] score {hit['score']:.3f} | {hit['law_title']}, მუხლი {hit['article']}. {hit['article_title']}")
        body = hit["text"].split("\n", 1)[-1]
        print("    " + body[:250].replace("\n", " ") + ("..." if len(body) > 250 else ""))


if __name__ == "__main__":
    main()