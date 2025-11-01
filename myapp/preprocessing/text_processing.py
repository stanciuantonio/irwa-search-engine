"""
Text preprocessing functions for document corpus.
Migrated from Part 1 notebook - maintains exact same logic.

Changes for Part 2:
- Added PID at root level in preprocess_document() for efficient indexing
- Added load_preprocessed_corpus() helper to convert list to dict
"""

import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


def extract_product_details(details):
    """
    Extracts only the descriptive values from structured product_details.
    Example input: [{"Color": "Blue"}, {"Material": "Cotton"}]
    Output: "Blue Cotton"
    """
    values = []
    for category in details:
        values.extend(v for v in category.values())
    return " ".join(values)


def build_terms(document):
    """
    Preprocess the document text (title + description + product_details extracted) removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    Argument:
    document -- a dictionary with 'title' and 'description' keys

    Returns:
    tokens - a list of tokens corresponding to the input text after the preprocessing
    """
    # 1. Stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # 2. Text
    text = document['title'] + ' ' + document['description'] + ' ' + extract_product_details(document['product_details'])
    text = text.lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    text = text.split(" ")
    text = [term for term in text if term not in stop_words]
    text = [term for term in text if term != '']
    text = [stemmer.stem(term) for term in text]

    return text


def preprocess_document(document):
    """
    Process the document
    1. Build the tokens of the document
    2. Build the metadata tokens of the document
    3. Build the original attributes of the document

    Returns:
        dict: Preprocessed document with structure:
            - pid: str - document identifier (for efficient indexing in Part 2)
            - searchable_text: list[str] - tokenized, stemmed text
            - metadata: dict - category, brand, seller, sub_category
            - original: dict - all original document fields
    """

    # 1. Searchable tokens
    tokens = build_terms(document)

    # 2. Metadata tokens
    metadata = {
        'category': document.get('category', '').lower().strip(),
        'sub_category': document.get('sub_category', '').lower().strip(),
        'brand': document.get('brand', '').lower().strip(),
        'seller': document.get('seller', '').lower().strip()
    }

    # 3. Original attributes
    original = {
        "pid": document["pid"],
        "title": document["title"],
        "description": document["description"],
        "brand": document["brand"],
        "category": document["category"],
        "sub_category": document["sub_category"],
        "product_details": document["product_details"],
        "seller": document["seller"],
        "out_of_stock": document["out_of_stock"],
        "selling_price": document["selling_price"],
        "discount": document["discount"],
        "actual_price": document["actual_price"],
        "average_rating": document["average_rating"],
        "url": document["url"]
    }
    return {
        "pid": document["pid"],  # PID at root level for O(1) access during indexing
        "searchable_text": tokens,
        "metadata": metadata,
        "original": original
    }


def load_preprocessed_corpus(cache_path='data/processed/preprocessed_corpus.pkl'):
    """
    Load preprocessed corpus from cache and convert to dict indexed by PID.

    This function is essential for Part 2 (Indexing) as it provides O(1) access
    to documents by their PID, which is needed when building the inverted index
    and retrieving documents for ranking.

    Args:
        cache_path: Path to preprocessed corpus pickle file

    Returns:
        dict: {pid: preprocessed_doc} for efficient document access

    Example:
        >>> corpus = load_preprocessed_corpus()
        >>> doc = corpus['TKPFCZ9EA7H5FYZH']  # O(1) access by PID
        >>> print(doc['searchable_text'][:5])
    """
    with open(cache_path, 'rb') as f:
        corpus_list = pickle.load(f)

    # Convert list to dict indexed by PID for O(1) access
    corpus_dict = {}
    for doc in corpus_list:
        # Handle both old format (pid in original) and new format (pid at root)
        pid = doc.get('pid') or doc['original']['pid']
        corpus_dict[pid] = doc

    return corpus_dict

def build_query_terms(query):
    """
    Preprocess the query text removing stop words, stemming,
    transforming in lowercase and return the tokens of the text.

    Argument:
    query -- a string representing the user query

    Returns:
    tokens - a list of tokens corresponding to the input query after the preprocessing
    """
    # 1. Stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # 2. Text
    text = query.lower()
    text = ''.join(char if char.isalnum() or char.isspace() else ' ' for char in text)
    text = text.split(" ")
    text = [term for term in text if term not in stop_words]
    text = [term for term in text if term != '']
    text = [stemmer.stem(term) for term in text]

    return text
