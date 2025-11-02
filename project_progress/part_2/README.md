# Part 2: Indexing and Evaluation

## Overview

Implementation of inverted index, TF-IDF ranking, and evaluation metrics for fashion e-commerce search engine.

## How to Run

### Prerequisites

```bash
pip install -r ../../requirements.txt
python -m nltk.downloader stopwords punkt
```

### Execute Notebook

```bash
cd project_progress/part_2/
jupyter notebook part_2.ipynb
```

### Run Cells in Order

1. Cells 1-2: Load preprocessed corpus
2. Cells 3-6: Execute 5 test queries
3. Cells 7-8: Execute validation queries
4. Cell 9: Evaluate metrics

## Files Structure

- `part_2.ipynb`: Main implementation and results
- `../../myapp/search/algorithms.py`: InvertedIndex + TFIDFRanker classes
- `../../myapp/search/evaluation.py`: SearchEvaluator + all metrics
- `../../PARTE2.md`: Detailed technical documentation

## Results Summary

- **Vocabulary:** 20,906 unique terms
- **Corpus:** 28,080 documents
- **Test Queries:** 5 queries defined
- **Validation Queries:** 2 queries evaluated
  - Query 1: P@10=0.200, R@10=0.154, MRR=0.333
  - Query 2: P@10=0.000, R@10=0.000, MRR=0.000 (critical issue)
- **Aggregate:** MAP@20=0.028, MRR=0.167

## Authors

Antonio Stanciu, Martí Girón

## Date

October 2025
