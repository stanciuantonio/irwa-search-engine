import math
import time
from collections import defaultdict
from myapp.preprocessing.text_processing import build_query_terms

class InvertedIndex:
    """
    Inverted index implementation based on part_2.ipynb approach.
    Uses term_index for numerical IDs and stores doc indices in sets.
    """

    def __init__(self, corpus):
        """
        Build inverted index from preprocessed corpus

        Args:
            corpus: dict {pid: preprocessed_doc}
        """
        self.corpus = corpus
        self.pid_list = list(corpus.keys())  # To convert doc_idx ↔ pid
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.pid_list)}

        # Extract vocabulary from corpus
        all_tokens = []
        for doc in corpus.values():
            all_tokens.extend(doc['searchable_text'])
        vocabulary = set(all_tokens)

        # Create term_index: {term: term_id}
        self.term_index = {term: i for i, term in enumerate(vocabulary)}

        # Build inverted index: {term_id: set(doc_indices)}
        self.index = defaultdict(set)

        start_time = time.time()
        for doc_idx, pid in enumerate(self.pid_list):
            doc = corpus[pid]
            tokens = doc['searchable_text']

            # Add document index to each term's posting list
            for term in tokens:
                term_id = self.term_index[term]
                self.index[term_id].add(doc_idx)

        self.build_time = time.time() - start_time
        self.total_docs = len(corpus)

        print(f"Indexed {len(self.term_index)} unique terms from {self.total_docs} documents in {self.build_time:.2f}s")

    def get_docs_with_term(self, term):
        """
        Return set of document indices containing the term

        Args:
            term: string term (not term_id)

        Returns:
            set of document indices
        """
        term_id = self.term_index.get(term)
        if term_id is None:
            return set()
        return self.index.get(term_id, set())

    def search_conjunctive(self, query_terms):
        """
        Conjunctive search (AND): find docs containing ALL query terms

        Args:
            query_terms: list of preprocessed query tokens

        Returns:
            set of document indices that contain all terms
        """
        docs = None

        for term in query_terms:
            term_docs = self.get_docs_with_term(term)

            if not term_docs:
                # If any term is not in index, no results
                return set()

            if docs is None:
                docs = term_docs.copy()
            else:
                docs.intersection_update(term_docs)

        return docs if docs else set()


class TFIDFRanker:
    """TF-IDF ranking algorithm"""

    def __init__(self, inverted_index):
        self.index = inverted_index
        self.corpus = inverted_index.corpus
        self.pid_list = inverted_index.pid_list
        self.total_docs = inverted_index.total_docs

    def calculate_tf(self, term, doc_tokens):
        """Term Frequency: count of term in document"""
        return doc_tokens.count(term)

    def calculate_idf(self, term):
        """Inverse Document Frequency"""
        docs_with_term = len(self.index.get_docs_with_term(term))
        if docs_with_term == 0:
            return 0
        return math.log(self.total_docs / docs_with_term)

    def calculate_tfidf(self, term, doc_tokens):
        """TF-IDF score for a term in a document"""
        tf = self.calculate_tf(term, doc_tokens)
        idf = self.calculate_idf(term)
        return tf * idf

    def rank_documents(self, query_terms, candidate_doc_indices):
        """
        Rank documents by TF-IDF score

        Args:
            query_terms: list of preprocessed query tokens
            candidate_doc_indices: set of document indices that match all query terms

        Returns:
            list of (pid, score) tuples sorted by score descending
        """
        scores = []

        for doc_idx in candidate_doc_indices:
            pid = self.pid_list[doc_idx]
            doc = self.corpus[pid]
            doc_tokens = doc['searchable_text']

            # Calculate TF-IDF score for this document
            doc_score = 0
            for term in query_terms:
                doc_score += self.calculate_tfidf(term, doc_tokens)

            scores.append((pid, doc_score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


def search_in_corpus(query, corpus=None, top_k=20):
    """
    Search documents using conjunctive query (AND) and TF-IDF ranking

    This function implements the complete pipeline:
    1. Preprocess query
    2. Build inverted index
    3. Find documents with ALL query terms (conjunctive/AND)
    4. Rank results by TF-IDF

    Args:
        query: string query from user
        corpus: preprocessed corpus dict (if None, loads from cache)
        top_k: number of results to return

    Returns:
        list of (pid, score) tuples
    """

    # Preprocess query
    query_terms = build_query_terms(query)
    print(f"Processed query terms: {query_terms}")

    if not query_terms:
        return []

    # Build inverted index
    inv_index = InvertedIndex(corpus)

    # Find documents containing ALL query terms (AND/conjunctive)
    candidate_doc_indices = inv_index.search_conjunctive(query_terms)

    # If no documents match all terms, return empty
    if not candidate_doc_indices:
        print("No documents found matching all query terms")
        return []

    print(f"Found {len(candidate_doc_indices)} documents matching all terms")

    # Rank candidates using TF-IDF
    ranker = TFIDFRanker(inv_index)
    ranked_results = ranker.rank_documents(query_terms, candidate_doc_indices)

    # Return top K results
    return ranked_results[:top_k]
