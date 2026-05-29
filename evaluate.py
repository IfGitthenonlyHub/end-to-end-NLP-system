import argparse
import collections
import re
import string
from pathlib import Path


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def answer_recall_score(prediction: str, ground_truth: str) -> float:
    pred = normalize_answer(prediction)
    truth = normalize_answer(ground_truth)
    if not truth:
        return float(not pred)
    return float(truth in pred)


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    return max(metric_fn(prediction, ground_truth) for ground_truth in ground_truths)


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate system outputs with EM, F1, and answer recall.")
    parser.add_argument("--predictions", default="system_outputs/system_output_1.txt")
    parser.add_argument("--references", default="data/test/reference_answers.txt")
    args = parser.parse_args()

    predictions = load_lines(Path(args.predictions))
    references = load_lines(Path(args.references))
    if len(predictions) != len(references):
        raise ValueError(f"Line count mismatch: {len(predictions)} predictions vs {len(references)} references")

    em = []
    f1 = []
    recall = []
    for prediction, reference_line in zip(predictions, references):
        ground_truths = [item.strip() for item in reference_line.split(";") if item.strip()]
        em.append(metric_max_over_ground_truths(exact_match_score, prediction, ground_truths))
        f1.append(metric_max_over_ground_truths(f1_score, prediction, ground_truths))
        recall.append(metric_max_over_ground_truths(answer_recall_score, prediction, ground_truths))

    count = len(predictions)
    print(f"Examples: {count}")
    print(f"Exact Match: {sum(em) / count * 100:.2f}")
    print(f"F1: {sum(f1) / count * 100:.2f}")
    print(f"Answer Recall: {sum(recall) / count * 100:.2f}")


if __name__ == "__main__":
    main()
