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
```text
.
├── build_index.py         # build inverted index
├── build_queries.py       # parse and preprocess queries
├── search_random_query.py # run search with random query
├── cisi_index/            # generated index files (ignored)
```

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
- Lee Jiho - [2j2h5](https://github.com/2j2h5)
- Choi Junwon - [junwon4158](https://github.com/junwon4158)
