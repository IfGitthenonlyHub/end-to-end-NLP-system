import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from chunking import format_chunk_for_prompt
from rag_config import (
    CHUNKS_FILE,
    DEFAULT_FEWSHOT_K,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
    EMBEDDING_MODEL,
    INDEX_FILE,
    OLLAMA_MODEL,
OLLAMA_URL,
    RERANKER_MODEL,
    SYSTEM_OUTPUT_DIR,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class VietnameseReranker:
    def __init__(self, model_name: str, max_length: int = 2304) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, pairs: list[tuple[str, str]], batch_size: int = 4) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.max_length,
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                scores.extend(logits.detach().cpu().numpy().tolist())
        return np.array(scores, dtype="float32")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_questions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_qa_pairs(questions_path: Path, answers_path: Path) -> list[dict]:
    if not questions_path.exists() or not answers_path.exists():
        return []
    questions = load_questions(questions_path)
    answers = load_questions(answers_path)
    return [{"question": q, "answer": a} for q, a in zip(questions, answers)]


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms


def retrieve(
    question: str,
    embedder: SentenceTransformer,
    index: faiss.Index,
    chunks: list[dict],
    top_k: int,
) -> list[dict]:
    query = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    scores, ids = index.search(query, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(chunks[int(idx)])
        item["retrieval_score"] = float(score)
        results.append(item)
    return results


def rerank(question: str, candidates: list[dict], reranker: VietnameseReranker, top_k: int) -> list[dict]:
    if not candidates:
        return []
    pairs = [(question, candidate["text"]) for candidate in candidates]
    scores = reranker.predict(pairs)
    ranked = []
    for candidate, score in zip(candidates, scores):
        item = dict(candidate)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
    return ranked[:top_k]


def precompute_train_question_embeddings(
    train_pairs: list[dict],
    embedder: SentenceTransformer,
) -> np.ndarray:
    if not train_pairs:
        return np.empty((0, 0), dtype="float32")
    train_questions = [pair["question"] for pair in train_pairs]
    return embedder.encode(
        train_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")


def select_fewshot_examples(
    question: str,
    train_pairs: list[dict],
    train_vectors: np.ndarray,
    query_vector: np.ndarray,
    k: int,
) -> list[dict]:
    if not train_pairs or k <= 0 or train_vectors.size == 0:
        return []
    scores = (train_vectors @ query_vector.T).reshape(-1)
    top_ids = np.argsort(scores)[::-1][:k]
    return [train_pairs[int(i)] for i in top_ids]


def build_prompt(question: str, contexts: list[dict], fewshots: list[dict]) -> str:
    context_text = "\n\n---\n\n".join(format_chunk_for_prompt(chunk) for chunk in contexts)
    examples = ""
    if fewshots:
        examples = "\n".join(
            f"Question: {item['question']}\nAnswer: {item['answer']}" for item in fewshots
        )
        examples = f"\nFew-shot examples:\n{examples}\n"

    return f"""You are a factual question-answering system for a RAG assignment.
Use only the provided context to answer the question.
Return only the shortest answer phrase.
Do not explain, do not cite sources, and do not write a full sentence unless necessary.
If the answer is not present in the context, return unknown.
Keep the answer in the same language as the question.
{examples}
Context:
{context_text}

Question: {question}
Answer:"""


def call_ollama(prompt: str, model: str, url: str, timeout: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.9,
            "num_ctx": 4096,
            "num_predict": 32,
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return clean_answer(data.get("response", ""))


def encode_query(embedder: SentenceTransformer, question: str) -> np.ndarray:
    return embedder.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def check_ollama_model(model: str, url: str, timeout: int) -> None:
    tags_url = url.rsplit("/", 1)[0] + "/tags"
    response = requests.get(tags_url, timeout=timeout)
    response.raise_for_status()
    models = response.json().get("models", [])
    names = {item.get("name") for item in models}
    if model not in names:
        available = ", ".join(sorted(name for name in names if name)) or "none"
        raise RuntimeError(
            f"Ollama model '{model}' is not installed. "
            f"Available models: {available}. "
            f"Run: ollama pull {model}"
        )


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(Answer|A|Trả lời|Đáp án)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.splitlines()[0].strip() if answer else "unknown"
    answer = answer.strip(" \"'`")
    return answer or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG inference over a questions.txt file.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--output", default=str(SYSTEM_OUTPUT_DIR / "system_output_1.txt"))
    parser.add_argument("--train-questions", default="data/train/questions.txt")
    parser.add_argument("--train-answers", default="data/train/reference_answers.txt")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--reranker-model", default=RERANKER_MODEL)
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--retrieve-top-k", type=int, default=DEFAULT_RETRIEVE_TOP_K)
    parser.add_argument("--rerank-top-k", type=int, default=DEFAULT_RERANK_TOP_K)
    parser.add_argument("--no-reranker", action="store_true", help="Use FAISS ranking directly without reranking.")
    parser.add_argument("--context-top-k", type=int, default=5, help="Number of FAISS chunks to pass when reranker is disabled.")
    parser.add_argument("--fewshot-k", type=int, default=DEFAULT_FEWSHOT_K)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true", help="Continue from an existing output file.")
    args = parser.parse_args()

    if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
        raise RuntimeError("FAISS index is missing. Run: python build_index.py")

    questions = load_questions(Path(args.questions))
    chunks = load_jsonl(CHUNKS_FILE)
    index = faiss.read_index(str(INDEX_FILE))
    train_pairs = load_qa_pairs(Path(args.train_questions), Path(args.train_answers))

    check_ollama_model(args.ollama_model, args.ollama_url, min(args.timeout, 30))

    print(f"Loading embedding model: {args.embedding_model}")
    embedder = SentenceTransformer(args.embedding_model, trust_remote_code=True)
    reranker = None
    if not args.no_reranker:
        print(f"Loading reranker model: {args.reranker_model}")
        reranker = VietnameseReranker(args.reranker_model)
    else:
        print("Reranker disabled; using FAISS ranking directly")
    print(f"Encoding {len(train_pairs)} train questions for few-shot selection")
    train_vectors = precompute_train_question_embeddings(train_pairs, embedder)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.resume and output_path.exists():
        outputs = output_path.read_text(encoding="utf-8").splitlines()
        outputs = outputs[: len(questions)]
        print(f"Resuming from {len(outputs)} existing answers")

    for i, question in enumerate(questions[len(outputs) :], start=len(outputs) + 1):
        query_vector = encode_query(embedder, question)
        scores, ids = index.search(query_vector, args.retrieve_top_k)
        candidates = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = dict(chunks[int(idx)])
            item["retrieval_score"] = float(score)
            candidates.append(item)
        if args.no_reranker:
            contexts = candidates[: args.context_top_k]
        else:
            contexts = rerank(question, candidates, reranker, args.rerank_top_k)
        fewshots = select_fewshot_examples(question, train_pairs, train_vectors, query_vector, args.fewshot_k)
        prompt = build_prompt(question, contexts, fewshots)
        print(f"[{i}/{len(questions)}] Generating answer")
        try:
            answer = call_ollama(prompt, args.ollama_model, args.ollama_url, args.timeout)
        except requests.RequestException as exc:
            print(f"[{i}/{len(questions)}] Ollama error: {exc}")
            answer = "unknown"
        outputs.append(answer)
        output_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
        print(f"[{i}/{len(questions)}] {question} -> {answer}")

    output_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
