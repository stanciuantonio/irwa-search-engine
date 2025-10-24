# Part 1: Text Processing and Exploratory Data Analysis

## Overview

This folder contains the implementation of Part 1 of the IRWA Search Engine project.

## Dataset

- **Source**: Fashion e-commerce products dataset
- **Total documents**: 28,080 products
- **Format**: JSON

## Files

- `part_1.ipynb`: Main Jupyter notebook with preprocessing and EDA
- `part1_report.pdf`: Detailed report explaining design decisions
- `README.md`: This file

## Preprocessing Approach

### Text Fields Processing

Text fields (title, description, product_details) are processed with:

1. **Lowercase normalization**
2. **Punctuation removal**
3. **Tokenization**
4. **Stop words removal** (using NLTK English stopwords)
5. **Stemming** (Porter Stemmer)

### Non-Text Fields Strategy: Hybrid Approach

**Metadata Fields** (brand, category, sub_category, seller):

- Stored as structured dictionary
- Converted to lowercase
- **NO stemming applied** (to preserve exact names for filtering)

**Numeric Fields** (price, rating, discount, out_of_stock):

- **NOT indexed as text**
- Stored in `original` for filtering and ranking
- Rationale: Numbers like "921" or "3.9" are meaningless for text search

### Processed Document Structure

Each processed document contains three components:

```python
{
    "searchable_text": [...],      # Preprocessed tokens with stemming
    "metadata": {                   # Structured fields without stemming
        "brand": "york",
        "category": "clothing and accessories",
        "sub_category": "bottomwear",
        "seller": "shyam enterprises"
    },
    "original": {                   # All original fields for display
        "pid": "...",
        "title": "...",
        "price": "...",
        # ... all other fields
    }
}
```

### Justification

Based on validation queries:

- Query 1: "women full sleeve sweatshirt cotton"
- Query 2: "men slim jeans blue"

Our hybrid approach allows:

- Text search in title/description/product_details
- Exact filtering by brand/category
- Numeric filtering and ranking by price/rating

## How to Run

```bash
# Install dependencies
pip install -r ../../requirements.txt

# Run notebook
jupyter notebook part_1.ipynb
```

## Dependencies

- Python 3.8+
- nltk
- pandas
- matplotlib
- seaborn
- numpy

## Authors

Antonio Stanciu and Martí Girón

## Date

October 2025
