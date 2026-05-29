import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import make_chunks
from evaluate import normalize_answer
from rag_config import EMBEDDING_MODEL, EXPERIMENT_DIR, KNOWLEDGE_DIR


def load_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer_in_chunks(answer_line: str, chunks: list[dict]) -> bool:
    answers = [normalize_answer(item.strip()) for item in answer_line.split(";") if item.strip()]
    merged = normalize_answer("\n".join(chunk["text"] for chunk in chunks))
    return any(answer and answer in merged for answer in answers)


def evaluate_top_k(
    questions: list[str],
    answers: list[str],
    chunks: list[dict],
    query_vectors: np.ndarray,
    index: faiss.Index,
    top_k: int,
) -> float:
    _, ids = index.search(query_vectors, top_k)

    hits = 0
    for answer, row in zip(answers, ids):
        retrieved = [chunks[int(i)] for i in row if i >= 0]
        hits += int(answer_in_chunks(answer, retrieved))
    return hits / len(questions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune chunk size and retrieval top-k on train data.")
    parser.add_argument("--train-questions", default="data/train/questions.txt")
    parser.add_argument("--train-answers", default="data/train/reference_answers.txt")
    parser.add_argument("--chunk-sizes", default="300,450,650")
    parser.add_argument("--overlap", type=int, default=80)
    parser.add_argument("--top-ks", default="5,10,20")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    args = parser.parse_args()

    questions = load_lines(Path(args.train_questions))
    answers = load_lines(Path(args.train_answers))
    if len(questions) != len(answers):
        raise ValueError("Train questions and answers must have the same number of lines.")

    chunk_sizes = [int(item) for item in args.chunk_sizes.split(",")]
    top_ks = [int(item) for item in args.top_ks.split(",")]
    embedder = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    query_vectors = embedder.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    results = []
    for chunk_size in chunk_sizes:
        chunks = [chunk.to_dict() for chunk in make_chunks(KNOWLEDGE_DIR, chunk_size, args.overlap)]
        chunk_vectors = embedder.encode(
            [chunk["text"] for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")
        index = faiss.IndexFlatIP(chunk_vectors.shape[1])
        index.add(chunk_vectors)
        for top_k in top_ks:
            recall_at_k = evaluate_top_k(questions, answers, chunks, query_vectors, index, top_k)
            result = {
                "chunk_size": chunk_size,
                "chunk_overlap": args.overlap,
                "top_k": top_k,
                "retrieval_answer_recall": recall_at_k,
                "num_chunks": len(chunks),
            }
            results.append(result)
            print(result)

    results.sort(key=lambda item: item["retrieval_answer_recall"], reverse=True)
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXPERIMENT_DIR / "retrieval_tuning.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved tuning results to {output_path}")
    print(f"Best setting: {results[0]}")


if __name__ == "__main__":
    main()
