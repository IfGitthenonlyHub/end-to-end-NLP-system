import argparse
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import make_chunks
from pathlib import Path
from rag_config import (
    CHUNKS_FILE,
    CONFIG_FILE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INDEX_FILE,
    KNOWLEDGE_DIR,
)


def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embeddings.astype("float32")


def save_chunks(chunks: list[dict]) -> None:
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS vector index for the RAG system.")
    parser.add_argument("--knowledge-dir", default=str(KNOWLEDGE_DIR))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = make_chunks(
        knowledge_dir=Path(args.knowledge_dir),
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )
    if not chunks:
        raise RuntimeError(f"No chunks found in {args.knowledge_dir}")

    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    save_chunks(chunk_dicts)

    model = load_embedding_model(args.embedding_model)
    embeddings = embed_texts(model, [chunk["text"] for chunk in chunk_dicts], args.batch_size)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))

    config = {
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "num_chunks": len(chunk_dicts),
        "index_type": "faiss.IndexFlatIP",
        "normalized_embeddings": True,
    }
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built index with {len(chunk_dicts)} chunks")
    print(f"FAISS index: {INDEX_FILE}")
    print(f"Chunk metadata: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
