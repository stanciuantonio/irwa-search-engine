import pickle
import pandas as pd

from myapp.search.objects import Document
from typing import List, Dict


def load_corpus(path) -> List[Document]:
    """
    Load file and transform to dictionary with each document as an object for easier treatment when needed for displaying
     in results, stats, etc.
    :param path:
    :return:
    """
    df = pd.read_json(path)
    corpus = _build_corpus(df)
    return corpus

def _build_corpus(df: pd.DataFrame) -> Dict[str, Document]:
    """
    Build corpus from dataframe
    :param df:
    :return:
    """
    corpus = {}
    for _, row in df.iterrows():
        doc = Document(**row.to_dict())
        corpus[doc.pid] = doc
    return corpus

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
