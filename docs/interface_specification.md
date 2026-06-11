# Interface Specification

---

## Overview

The system consists of interchangeable retrieval components that follow common interface contracts.

Core components:

- Tokenizer
- InvertedIndex
- Weighter
- IntentEstimator
- RetrievalModel
- Evaluator

The purpose of these interfaces is to allow multiple retrieval approaches (Boolean, VSM, Link-VSM, Intent-VSM) to share the same infrastructure.

---

## Tokenizer

### Purpose

Convert raw text into normalized tokens.

### Interface

```python
tokenize(text: str) -> list[str]
```

### Notes

- Used for both documents and queries
- Must apply consistent preprocessing

---

## InvertedIndex

### Purpose

Store and retrieve document statistics.

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

- Field-aware indexing
- Supports title and body statistics

---

## Weighter

### Purpose

Convert term frequencies into weighted vectors.

### Interface

```python
document_vector(
    doc_id: int,
    index
) -> dict[str, float]

query_vector(
    query_term_counts: dict[str, int],
    index
) -> dict[str, float]
```

### Notes

Examples:

- TF-IDF
- Future BM25 extensions

---

## IntentEstimator

### Purpose

Estimate user intent from semantic query relationships.

### Interface

```python
build() -> None

embed_query(
    query: str
)

retrieve_similar_queries(
    query: str,
    top_k: int
)

construct_intent_vector(
    query: str
)
```

### Notes

- Used only by Intent-VSM
- May use sentence embeddings
- May incorporate neighboring query information

---

## RetrievalModel

### Purpose

Retrieve documents for a given query.

### Interface

```python
build() -> None

search(
    query: str,
    top_k: int = 10
) -> list[tuple[int, float]]
```

### Optional Interface

```python
explain(
    query: str,
    doc_id: int
) -> dict[str, float]
```

### Notes

Supported implementations:

- Boolean Model
- Vector Space Model
- Link-VSM
- Intent-VSM

---

## Evaluator

### Purpose

Evaluate retrieval effectiveness.

### Interface

```python
evaluate_query(
    query: str,
    relevant_docs: set[int],
    k: int = 10
) -> dict[str, float]
```

```python
evaluate_all(
    queries: dict[int, str],
    relevance: dict[int, set[int]],
    k: int = 10
) -> dict[str, any]
```

### Notes

Supported metrics:

- Precision
- Recall
- F-score
- Average Precision
- Mean Average Precision

All retrieval models must expose a compatible search interface.
