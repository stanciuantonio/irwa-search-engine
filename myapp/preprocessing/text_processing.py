"""
Text preprocessing functions for document corpus.
Migrated from Part 1 notebook - maintains exact same logic.
"""

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
        "searchable_text": tokens,
        "metadata": metadata,
        "original": original
    }
