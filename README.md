# CISI IR System

## Overview

This project implements a basic Information Retrieval(IR) System using the CISI dataset.
It builds an inverted index and retrieves relevant documents based on term freqyency and rarity-based scoring.

## What This Project Does

- Builds an inverted index from documents
- Processes queries from the dataset
- Scores documents using TF + rarity weighting
- Retrieves top-3 relevant documents
- Shows term contribution for each result

## Dataset
- [CISI Dataset (Kaggle)](https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval)

## How to Run
```bash
python build_index.py
python build_queries.py
python search_random_query.py
```

## Project Structure
.
├── build_index.py
├── build_queries.py
├── search_random_query.py
├── cisi_index/   # ignored (generated files)

## Result & Analysis
- Relevant documents are generally retrieved successfully
- However, some terms dominate the scoring
- Repeated query terms are counted multiple times
- The system is purely keyword-based and lacks semantic understanding

## Future Work
- TF-IDF with log scaling
- BM25 ranking
- Stopword removal
- Query expansion
- Semantic search

## Author
- Lee Jiho
- Choi Junwon
