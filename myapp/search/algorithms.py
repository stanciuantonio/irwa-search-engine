import math
import time
from collections import defaultdict

from myapp.preprocessing.text_processing import build_query_terms

class InvertedIndex:
    """
    Inverted index + TF-IDF document vectors.

    - Builds postings lists (term -> documents)
    - Computes IDF per term
    - Precomputes normalized TF-IDF vectors for each document
    """

    def __init__(self, corpus):
        """
        Build inverted index and TF-IDF document vectors from preprocessed corpus.

        Args:
            corpus: dict {pid: preprocessed_doc}
        """
        self.corpus = corpus
        self.pid_list = list(corpus.keys())  # To convert doc_idx ↔ pid
        self.pid_to_idx = {pid: idx for idx, pid in enumerate(self.pid_list)}

        start_time = time.time()

        # Core structures
        self.term_index = {}  # term -> term_id
        self.index = defaultdict(set)  # term_id -> {doc_idx}
        self.doc_tf = {}  # doc_idx -> {term_id: tf}
        self.idf = {}  # term_id -> idf
        self.doc_tfidf = {}  # doc_idx -> {term_id: weight}
        self.doc_norms = {}  # doc_idx -> ||d||

        # 1) Assign term IDs and count term frequencies per document
        for doc_idx, pid in enumerate(self.pid_list):
            tokens = corpus[pid]["searchable_text"]
            term_counts = defaultdict(int)

            for term in tokens:
                if term not in self.term_index:
                    self.term_index[term] = len(self.term_index)
                term_id = self.term_index[term]
                term_counts[term_id] += 1

            self.doc_tf[doc_idx] = term_counts

        self.total_docs = len(self.pid_list)

        # 2) Build postings and document frequencies
        df_counts = defaultdict(int)
        for doc_idx, term_counts in self.doc_tf.items():
            for term_id in term_counts.keys():
                self.index[term_id].add(doc_idx)

        for term_id, posting in self.index.items():
            df_counts[term_id] = len(posting)

        # 3) Compute IDF for each term
        for term_id, df in df_counts.items():
            self.idf[term_id] = math.log(self.total_docs / df) if df > 0 else 0.0

        # 4) Precompute TF-IDF vectors and norms for each document
        for doc_idx, term_counts in self.doc_tf.items():
            vec = {}
            norm_sq = 0.0
            for term_id, tf in term_counts.items():
                weight = tf * self.idf[term_id]
                vec[term_id] = weight
                norm_sq += weight * weight

            self.doc_tfidf[doc_idx] = vec
            self.doc_norms[doc_idx] = math.sqrt(norm_sq) if norm_sq > 0 else 0.0

        self.build_time = time.time() - start_time

        print(
            f"Indexed {len(self.term_index)} unique terms from "
            f"{self.total_docs} documents in {self.build_time:.2f}s"
        )

    def get_docs_with_term(self, term):
        """
        Return set of document indices containing the term.

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
        Conjunctive search (AND): find docs containing ALL query terms.

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
    """
    TF-IDF ranking algorithm using cosine similarity between
    query and document TF-IDF vectors.
    """

    def __init__(self, inverted_index: InvertedIndex):
        self.index = inverted_index
        self.corpus = inverted_index.corpus
        self.pid_list = inverted_index.pid_list
        self.total_docs = inverted_index.total_docs

    def _build_query_vector(self, query_terms):
        """
        Build TF-IDF vector for the query and compute its norm.
        """
        term_counts = defaultdict(int)
        for term in query_terms:
            term_id = self.index.term_index.get(term)
            if term_id is not None:
                term_counts[term_id] += 1

        if not term_counts:
            return {}, 0.0

        vec = {}
        norm_sq = 0.0
        for term_id, tf in term_counts.items():
            idf = self.index.idf.get(term_id, 0.0)
            weight = tf * idf
            vec[term_id] = weight
            norm_sq += weight * weight

        norm = math.sqrt(norm_sq) if norm_sq > 0 else 0.0
        return vec, norm

    @staticmethod
    def _cosine_similarity(q_vec, q_norm, d_vec, d_norm):
        """
        Compute cosine similarity between query and document vectors.
        """
        if q_norm == 0 or d_norm == 0:
            return 0.0

        dot = 0.0
        for term_id, q_weight in q_vec.items():
            d_weight = d_vec.get(term_id)
            if d_weight is not None:
                dot += q_weight * d_weight

        return dot / (q_norm * d_norm)

    def rank_documents(self, query_terms, candidate_doc_indices):
        """
        Rank documents by TF-IDF cosine similarity.

        Args:
            query_terms: list of preprocessed query tokens
            candidate_doc_indices: set of document indices that match all query terms

        Returns:
            list of (pid, score) tuples sorted by score descending
        """
        q_vec, q_norm = self._build_query_vector(query_terms)
        if not q_vec:
            return []

        scores = []

        for doc_idx in candidate_doc_indices:
            pid = self.pid_list[doc_idx]
            d_vec = self.index.doc_tfidf[doc_idx]
            d_norm = self.index.doc_norms[doc_idx]
            score = self._cosine_similarity(q_vec, q_norm, d_vec, d_norm)
            scores.append((pid, score))

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
