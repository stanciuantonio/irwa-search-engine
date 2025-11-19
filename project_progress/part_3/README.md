# Part 3: Ranking and Filtering

## Overview

Implementation and comparison of several ranking functions for the fashion e‑commerce search engine:

- TF‑IDF + cosine similarity
- BM25
- Custom Score (BM25 + rating + discount + stock)
- Word2Vec + cosine similarity
- FastText + cosine similarity (extra)

All models are tested on the 5 queries defined in Part 2 using our own relevance labels.

## How to Run

### Prerequisites

From the project root:

```bash
pip install -r requirements.txt
```

### Execute Notebook

```bash
cd project_progress/part_3/
jupyter notebook part_3.ipynb
```

### Run Cells in Order

1. Import & load preprocessed corpus.
2. **TF‑IDF** section:
   - Run the test query.
   - Run the 5 test queries.
3. **BM25** section:
   - Run the test query.
   - Run the 5 test queries.
4. **Custom Score** section:
   - Run the test query.
   - Run the 5 test queries.
5. **Word2Vec** section:
   - Train or load the Word2Vec model.
   - Build embeddings.
   - Run the test query and the 5 test queries.
6. **FastText (extra)** section:
   - Train or load the FastText model.
   - Build embeddings.
   - Run the test query and the 5 test queries.
7. **Metrics & comparison** section:
   - Load `data/test_queries_labels.csv`.
   - Compute P@10, R@10, AP@20, MRR, NDCG@10 for all models.
   - Inspect comparison tables (TF‑IDF vs BM25 vs Custom vs Word2Vec vs FastText, and Word2Vec vs FastText).

## Files Structure

- `part_3.ipynb`: main implementation and experiments for Part 3.
- `README.md`: this file.
- `../../myapp/search/algorithms.py`:
  - `InvertedIndex`, `TFIDFRanker`, `BM25Ranker`,
  - `CustomScoreRanker`, `Word2VecRanker`.
- `../../myapp/search/search_engine.py`:
  - `search_tfidf`, `search_bm25`, `search_custom`, `search_word2vec`.
- `../../myapp/core/scoring_utils.py`:
  - helpers to parse/normalize rating, discount, price.
- `../../myapp/search/evaluation.py`:
  - `SearchEvaluator` and all evaluation metrics.
- `../../data/processed/preprocessed_corpus.pkl`:
  - preprocessed corpus.
- `../../data/test_queries_labels.csv`:
  - relevance labels for our 5 test queries.

## Results Summary (high level)

- **TF‑IDF vs BM25**

  - BM25 generally behaves similarly or slightly better than TF‑IDF when document length matters.
  - For some queries TF‑IDF ranks all relevant items at the very top; this is visible in the metrics table.

- **Custom Score (BM25 + rating + discount + stock)**

  - Maintains BM25‑level metrics on our labeled queries.
  - Re‑orders results to favor higher rating and larger discounts, while pushing down out‑of‑stock items.

- **Word2Vec & FastText**
  - Provide semantic rankings based on averaged word embeddings.
  - Word2Vec performs well when query terms are present in the training vocabulary.
  - FastText, thanks to character n‑grams, is more robust to rare words and typos and achieves similar quantitative performance.

Exact metric values (P@10, R@10, AP@20, MRR, NDCG@10) per query and per model are reported in the final cells of `part_3.ipynb`.

## Authors

Antonio Stanciu, Martí Girón

## Date

November 2025
