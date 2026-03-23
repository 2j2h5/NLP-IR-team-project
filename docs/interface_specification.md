# Interface Specification

---

## Overview

The system is composed of the following interchangeable components:

- Tokenizer
- InvertedIndex
- Weighter
- RetrievalModel
- Evaluator

Each component follows a minimal interface contract.

---

## Tokenizer

### Purpose

Convert raw text into tokens.

### Interface

```python
tokenize(text: str) -> list[str]
```

### Notes

- Used in both indexing and query processing
- Must apply the same normalization for documents and queries

---

## InvertedIndex

### Purpose

Store and retrieve term-based document information.

### Interface

```python
get_postings(term: str) -> dict[int, dict[str, int]]
get_df(term: str) -> int

get_title_tf(term: str, doc_id: int) -> int
get_body_tf(term: str, doc_id: int) -> int

vocabulary() -> list[str]
get_document(doc_id: int) -> dict[str, str]
```

### Notes

- Provides all data required for weighting and retrieval
- Field-aware (title/body)

---

## Weighter

### Purpose

Convert term frequencies into weighted vectors.

### Interface

```python
document_vector(doc_id: int, index) -> dict[str, float]
query_vector(query_term_counts: dict[str, int], index) -> dict[str, float]
```

### Notes

- Used in ranked retrieval models only
- Examples: TF-IDF, BM25

---

## RetrievalModel

### Purpose

Retrieve documents for a given query.

### Interface

```python
build() -> None
search(query: str, top_k: int = 10) -> list[tuple[int, float]]
```

### Optional Interface

```python
explain(query: str, doc_id: int) -> dict[str, float]
```

### Notes

- `build()` prepares internal structures (e.g., document vectors)
- `search()` must return ranked results
- All models must support the same search interface

---

## Evaluator

### Purpose

Evaluate retrieval performance.

### Interface

```python
evaluate_query(
    query: str,
    relevant_docs: set[int],
    k: int = 10
) -> dict[str, float]

evaluate_all(
    queries: dict[int, str],
    relevance: dict[int, set[int]],
    k: int = 10
) -> dict[str, any]
```

### Notes
- Assumes the model implements `search(query, top_k)`
- Computes metrics such as Precision, Recall, MAP
