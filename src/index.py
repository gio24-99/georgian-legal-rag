"""Embed all law chunks with an Ollama embedding model and store them in ChromaDB.

    python src/index.py                      # default: bge-m3
    python src/index.py --embed_model nomic-embed-text   # (later: compare embedding models)

Each embedding model gets its own collection, so you can compare them in the evaluation.
"""
import argparse
import json
import time

import chromadb
import ollama

CHROMA_DIR = "data/chroma"


def collection_name(embed_model: str) -> str:
    return "laws_" + embed_model.replace(":", "_").replace("/", "_").replace(".", "_")


def get_collection(client, name: str, reset: bool = False):
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    # cosine similarity; the newer Chroma API uses `configuration`, older uses metadata
    try:
        return client.get_or_create_collection(name, configuration={"hnsw": {"space": "cosine"}})
    except TypeError:
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})


def embed(texts: list[str], model: str) -> list[list[float]]:
    return ollama.embed(model=model, input=texts)["embeddings"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/chunks.jsonl")
    p.add_argument("--embed_model", default="bge-m3")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    with open(args.chunks, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = get_collection(client, collection_name(args.embed_model), reset=True)

    start = time.time()
    for i in range(0, len(chunks), args.batch_size):
        batch = chunks[i : i + args.batch_size]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embed([c["text"] for c in batch], args.embed_model),
            documents=[c["text"] for c in batch],
            metadatas=[{k: c[k] for k in ("law_id", "law_title", "chapter", "article", "article_title")} for c in batch],
        )
        print(f"\r{min(i + args.batch_size, len(chunks))}/{len(chunks)}", end="", flush=True)
    print(f"\nIndexed {col.count()} chunks with {args.embed_model} in {time.time() - start:.1f}s "
          f"-> collection '{col.name}' in {CHROMA_DIR}")


if __name__ == "__main__":
    main()