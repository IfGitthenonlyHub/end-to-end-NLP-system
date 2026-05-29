# End-to-End NLP System Building: RAG QA

This repository implements a retrieval augmented generation (RAG) system for factual question answering. The current system follows the assignment pipeline:

1. Collect public documents from websites and PDFs.
2. Clean documents into `data/knowledgebase`.
3. Annotate QA pairs into train/test files.
4. Build a FAISS vector index over structurally chunked documents.
5. Retrieve candidate chunks, rerank them, and generate short answers with an Ollama LLM.
6. Write system outputs and evaluate them with Exact Match, F1, and answer recall.

## Models

- Embedding model: `AITeamVN/Vietnamese_Embedding`
- Vector index: FAISS
- Reranker: `AITeamVN/Vietnamese_Reranker`
- Generator: `qwen2.5:3b-instruct-q6_K` through Ollama

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install/pull the Ollama model:

```bash
ollama pull qwen2.5:3b-instruct-q6_K
```

Make sure Ollama is running:

```bash
ollama serve
```

## Data Preparation

The existing data scripts are:

```bash
python crawling.py
python cleaning.py
python auto-annotation.py
python split.py
```

The repository already contains generated knowledge base files and train/test QA files.

## Tune Retrieval on Train Data

Use train data as development data to choose chunk size and retrieval `top_k`:

```bash
python tune_retrieval.py --chunk-sizes 300,450,650 --top-ks 5,10,20
```

This writes:

```text
experiments/retrieval_tuning.json
```

## Build FAISS Index

Build the vector index with the selected structural chunking configuration:

```bash
python build_index.py --chunk-size 650 --chunk-overlap 80
```

This writes:

```text
data/index/faiss.index
data/index/chunks.jsonl
data/index/index_config.json
```

## Run RAG

Generate answers for the test questions:

```bash
python run_rag.py ^
  --questions data/test/questions.txt ^
  --output system_outputs/system_output_1.txt ^
  --retrieve-top-k 20 ^
  --rerank-top-k 3 ^
  --fewshot-k 3 ^
  --resume
```

The system uses train QA pairs as few-shot prompt examples. The output file contains one answer per line.

The current train-data tuning result selected:

```text
chunk_size=650
chunk_overlap=80
retrieve_top_k=20
```

## Run No-RAG Baseline

Generate a closed-book baseline without retrieval context:

```bash
python run_no_rag.py ^
  --questions data/test/questions.txt ^
  --output system_outputs/system_output_3.txt ^
  --resume
```

This file is useful for the report analysis comparing closed-book generation against RAG.

## Run RAG Without Reranker

Generate an experiment output using FAISS ranking directly:

```bash
python run_rag.py ^
  --questions data/test/questions.txt ^
  --output system_outputs/system_output_no_reranker.txt ^
  --retrieve-top-k 5 ^
  --context-top-k 5 ^
  --no-reranker ^
  --fewshot-k 3
```

## Evaluate

Evaluate against the local test references:

```bash
python evaluate.py ^
  --predictions system_outputs/system_output_2.txt ^
  --references data/test/reference_answers.txt
```

Metrics:

- Exact Match
- F1
- Answer Recall

## Submission Files

The assignment submission should include:

- `report.pdf`
- `github_url.txt`
- `contributions.md`
- `data/train/questions.txt`
- `data/train/reference_answers.txt`
- `data/test/questions.txt`
- `data/test/reference_answers.txt`
- `system_outputs/system_output_1.txt`
- `system_outputs/system_output_2.txt` is the current best local output from the tuned index.
- `system_outputs/system_output_3.txt` is the no-RAG closed-book baseline.
- `README.md`

## Notes

The generator prompt forces concise answers because the assignment metrics are token-based. Short answer phrases usually improve Exact Match and F1 compared with full-sentence explanations.
