import argparse
import re
import sys
from pathlib import Path

import requests

from rag_config import OLLAMA_MODEL, OLLAMA_URL, SYSTEM_OUTPUT_DIR
from run_rag import check_ollama_model


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_questions(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_answer(answer: str) -> str:
    answer = answer.strip()
    answer = re.sub(r"^(Answer|A|Trả lời|Đáp án)\s*:\s*", "", answer, flags=re.IGNORECASE)
    answer = answer.splitlines()[0].strip() if answer else "unknown"
    answer = answer.strip(" \"'`")
    return answer or "unknown"


def build_prompt(question: str) -> str:
    return f"""You are a factual question-answering system.
Answer the question from your own model knowledge only.
Return only the shortest answer phrase.
Do not explain, do not cite sources, and do not write a full sentence unless necessary.
If you do not know the answer, return unknown.
Keep the answer in the same language as the question.

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
            "num_ctx": 2048,
            "num_predict": 32,
        },
    }
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return clean_answer(response.json().get("response", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a closed-book no-RAG Ollama baseline.")
    parser.add_argument("--questions", default="data/test/questions.txt")
    parser.add_argument("--output", default=str(SYSTEM_OUTPUT_DIR / "system_output_3.txt"))
    parser.add_argument("--ollama-model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    check_ollama_model(args.ollama_model, args.ollama_url, min(args.timeout, 30))

    questions = load_questions(Path(args.questions))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = []
    if args.resume and output_path.exists():
        outputs = output_path.read_text(encoding="utf-8").splitlines()
        outputs = outputs[: len(questions)]
        print(f"Resuming from {len(outputs)} existing answers")

    for i, question in enumerate(questions[len(outputs) :], start=len(outputs) + 1):
        print(f"[{i}/{len(questions)}] Generating no-RAG answer")
        try:
            answer = call_ollama(build_prompt(question), args.ollama_model, args.ollama_url, args.timeout)
        except requests.RequestException as exc:
            print(f"[{i}/{len(questions)}] Ollama error: {exc}")
            answer = "unknown"
        outputs.append(answer)
        output_path.write_text("\n".join(outputs) + "\n", encoding="utf-8")
        print(f"[{i}/{len(questions)}] {question} -> {answer}")

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
