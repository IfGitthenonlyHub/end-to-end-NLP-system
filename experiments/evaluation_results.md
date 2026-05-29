# Evaluation Results

## Retrieval Tuning on Train Data

Embedding model: `AITeamVN/Vietnamese_Embedding`

Structural chunking overlap: `80`

| Chunk size | Top-k | Chunks | Train retrieval answer recall |
|---:|---:|---:|---:|
| 650 | 20 | 242 | 82.35 |
| 300 | 20 | 450 | 80.39 |
| 450 | 20 | 312 | 80.39 |
| 650 | 10 | 242 | 77.45 |
| 300 | 10 | 450 | 76.47 |
| 450 | 10 | 312 | 75.49 |
| 650 | 5 | 242 | 72.55 |
| 450 | 5 | 312 | 71.57 |
| 300 | 5 | 450 | 70.59 |

Selected setting:

```text
chunk_size=650
chunk_overlap=80
retrieve_top_k=20
rerank_top_k=3
fewshot_k=3
```

## Local Test Results

| Output file | Index setting | Exact Match | F1 | Answer Recall |
|---|---|---:|---:|---:|
| `system_outputs/system_output_1.txt` | chunk size 450, top-k 20, rerank top-k 3 | 56.00 | 74.72 | 72.00 |
| `system_outputs/system_output_2.txt` | chunk size 650, top-k 20, rerank top-k 3 | 60.00 | 83.13 | 84.00 |

Current best local output: `system_outputs/system_output_2.txt`.

## No-RAG Baseline

The no-RAG baseline was generated with `run_no_rag.py`, using only `qwen2.5:3b-instruct-q6_K` and no retrieved context.

| Output file | Method | Exact Match | F1 | Answer Recall |
|---|---|---:|---:|---:|
| `system_outputs/system_output_3.txt` | Closed-book Qwen2.5 no-RAG | 12.00 | 20.34 | 12.00 |

Note: `system_output_3.txt` was generated for the current `data/test/questions.txt`. If `data/test/questions.txt` is changed or regenerated, all system outputs must be regenerated so line-by-line evaluation remains valid.

## Updated 102-Question Test Set Results

Dataset:

```text
test questions: 102
train questions: 311
```

Index:

```text
chunk_size=650
chunk_overlap=80
num_chunks=242
embedding_model=AITeamVN/Vietnamese_Embedding
vector_index=FAISS IndexFlatIP
generator=qwen2.5:3b-instruct-q6_K via Ollama
```

| Output file | Method | Key configuration | Exact Match | F1 | Answer Recall |
|---|---|---|---:|---:|---:|
| `system_outputs/system_output_1.txt` | RAG + reranker | top-k 10, rerank top-3, few-shot 2 | 23.53 | 34.59 | 24.51 |
| `system_outputs/system_output_2.txt` | RAG + reranker | top-k 6, rerank top-2, few-shot 0 | 15.69 | 23.84 | 16.67 |
| `system_outputs/system_output_3.txt` | No-RAG closed-book | no retrieval context | 1.96 | 6.56 | 1.96 |

Current best output on the updated local test set: `system_outputs/system_output_1.txt`.

## Restored 25-Question Test Set: No-Reranker Experiment

Dataset:

```text
test questions: 25
train questions: 100
```

| Output file | Method | Key configuration | Exact Match | F1 | Answer Recall |
|---|---|---|---:|---:|---:|
| `system_outputs/system_output_no_reranker.txt` | RAG without reranker | top-k 5, context top-5, few-shot 3 | 32.00 | 52.37 | 40.00 |

Note: `system_outputs/system_output_2.txt` is not aligned with this restored test set in the current workspace state, so it should not be evaluated against the current `data/test/reference_answers.txt` unless restored/regenerated for the same question order.

## Restored 25-Question Test Set: Full RAG Regeneration

| Output file | Method | Key configuration | Exact Match | F1 | Answer Recall |
|---|---|---|---:|---:|---:|
| `system_outputs/system_output_2.txt` | RAG + reranker | top-k 20, rerank top-3, few-shot 3, num_ctx 4096, num_predict 32 | 60.00 | 74.90 | 64.00 |
| `system_outputs/system_output_no_reranker.txt` | RAG without reranker | top-k 5, context top-5, few-shot 3 | 32.00 | 52.37 | 40.00 |

Current best output on the restored 25-question test set: `system_outputs/system_output_2.txt`.
