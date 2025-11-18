import math
import time
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

from myapp.preprocessing.text_processing import build_query_terms
from myapp.core.scoring_utils import rating_norm, discount_norm

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
        # For BM25
        self.doc_lengths = {}  # doc_idx -> length
        total_length = 0
        self.avg_doc_length = 0  # average document length

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

            # For BM25
            doc_len = len(tokens)
            self.doc_lengths[doc_idx] = doc_len
            total_length += doc_len

        self.total_docs = len(self.pid_list)
        self.avg_doc_length = total_length / self.total_docs if self.total_docs > 0 else 0

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

class BM25Ranker:
    """
    BM25 ranking algorithm.

    Uses:
    - term frequencies from InvertedIndex.doc_tf
    - IDF from InvertedIndex.idf
    - document lengths and average length
    """

    def __init__(self, inverted_index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = inverted_index
        self.corpus = inverted_index.corpus
        self.pid_list = inverted_index.pid_list
        self.total_docs = inverted_index.total_docs
        self.k1 = k1
        self.b = b
        self.doc_lengths = inverted_index.doc_lengths
        self.avg_doc_length = inverted_index.avg_doc_length
        self.idf = inverted_index.idf

    def score_doc(self, query_terms, doc_idx: int) -> float:
        """
        Compute BM25 score for a single document and query.
        """
        score = 0.0
        doc_len = self.doc_lengths.get(doc_idx, 0)
        if doc_len == 0 or self.avg_doc_length == 0:
            return 0.0

        term_freqs = self.index.doc_tf[doc_idx]

        for term in query_terms:
            term_id = self.index.term_index.get(term)
            if term_id is None:
                continue

            tf = term_freqs.get(term_id, 0)
            if tf == 0:
                continue

            idf = self.idf.get(term_id, 0.0)

            # BM25 core formula
            numerator = tf * (self.k1 + 1.0)
            denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_length))
            score += idf * (numerator / denominator)

        return score

    def rank_documents(self, query_terms, candidate_doc_indices):
        """
        Rank documents by BM25 score.

        Args:
            query_terms: list of preprocessed query tokens
            candidate_doc_indices: set of document indices (AND semantics)

        Returns:
            list of (pid, score) tuples sorted by score descending
        """
        if not candidate_doc_indices:
            return []

        scores = []
        for doc_idx in candidate_doc_indices:
            pid = self.pid_list[doc_idx]
            score = self.score_doc(query_terms, doc_idx)
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


class CustomScoreRanker:
    """
    Custom ranking function that combines a textual BM25 score with
    product-level signals: rating, discount, and stock availability.

    your_score = alpha * bm25_score + beta  * rating_norm + gamma * discount_norm - delta * out_of_stock_penalty
    """

    def __init__(
        self,
        base_ranker: BM25Ranker,
        corpus,
        alpha: float = 0.7,
        beta: float = 0.2,
        gamma: float = 0.1,
        delta: float = 0.5,
    ):
        """
        Args:
            base_ranker: BM25Ranker instance built on the same InvertedIndex
            corpus: preprocessed corpus dict {pid: preprocessed_doc}
            alpha: weight for BM25 textual score
            beta: weight for normalized rating
            gamma: weight for normalized discount
            delta: penalty weight for out-of-stock items
        """
        self.base_ranker = base_ranker
        self.index = base_ranker.index
        self.corpus = corpus
        self.pid_list = self.index.pid_list

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _compute_custom_score(self, query_terms, doc_idx: int) -> float:
        """
        Compute the combined score for a given document.
        """
        pid = self.pid_list[doc_idx]
        doc = self.corpus[pid]
        original = doc["original"]

        # 1. Textual score from BM25
        bm25_score = self.base_ranker.score_doc(query_terms, doc_idx)

        # 2. Normalized numerical signals
        r_norm = rating_norm(original)
        d_norm = discount_norm(original)
        out_of_stock = bool(original.get("out_of_stock", False))
        out_penalty = 1.0 if out_of_stock else 0.0

        # 3. Linear combination
        return self.alpha * bm25_score + self.beta * r_norm + self.gamma * d_norm - self.delta * out_penalty

    def rank_documents(self, query_terms, candidate_doc_indices):
        """
        Rank documents by the custom combined score.

        Args:
            query_terms: list of preprocessed query tokens
            candidate_doc_indices: set of document indices (AND semantics)

        Returns:
            list of (pid, score) tuples sorted by score descending.
        """
        if not candidate_doc_indices:
            return []

        scores = []
        for doc_idx in candidate_doc_indices:
            pid = self.pid_list[doc_idx]
            score = self._compute_custom_score(query_terms, doc_idx)
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

class Word2VecRanker:
    """
    Ranker that represents documents and queries by the average of their
    word2vec word vectors, and ranks by cosine similarity.

    Args:
        inverted_index: instance of InvertedIndex (provides corpus and pid_list)
        embeddings: dict mapping token -> vector (iterable of floats, length = dim)
            e.g. {'apple': [0.12, -0.03, ...], ...}
        normalize_doc_vectors: if True, pre-normalize document vectors to unit length
    """

    def __init__(self, inverted_index, embeddings: Dict[str, Iterable[float]], normalize_doc_vectors: bool = True):
        self.index = inverted_index
        self.corpus = inverted_index.corpus
        self.pid_list = inverted_index.pid_list
        self.embeddings = embeddings
        # Determine embedding dimensionality from the first embedding
        any_vec = next(iter(embeddings.values())) if embeddings else None
        self.dim = len(any_vec) if any_vec is not None else 0

        # Precompute averaged document vectors (doc_idx -> vector list)
        self.doc_vectors = {}       # doc_idx -> list[float]
        self.doc_norms = {}         # doc_idx -> float (norm)
        self.normalize_doc_vectors = normalize_doc_vectors
        self._build_doc_vectors()

    def _vector_add_inplace(self, acc, vec):
        for i in range(len(acc)):
            acc[i] += vec[i]

    def _vector_scale_inplace(self, acc, s):
        for i in range(len(acc)):
            acc[i] *= s

    def _vec_dot(self, a, b):
        # assume same length
        dot = 0.0
        for i in range(len(a)):
            dot += a[i] * b[i]
        return dot

    def _vec_norm(self, v):
        s = 0.0
        for x in v:
            s += x * x
        return math.sqrt(s)

    def _build_doc_vectors(self):
        """
        For every document in the inverted index corpus, compute the average
        of word vectors for tokens present in corpus[pid]['searchable_text'].
        """
        for doc_idx, pid in enumerate(self.pid_list):
            tokens = self.corpus[pid].get("searchable_text", [])
            if not tokens:
                # zero vector
                vec = [0.0] * self.dim
                self.doc_vectors[doc_idx] = vec
                self.doc_norms[doc_idx] = 0.0
                continue

            acc = [0.0] * self.dim
            count = 0
            for t in tokens:
                v = self.embeddings.get(t)
                if v is None:
                    continue
                # support any iterable (tuple/list/ndarray-like)
                # ensure length matches expected dim; if not, skip
                if len(v) != self.dim:
                    continue
                if isinstance(v, (list, tuple)):
                    self._vector_add_inplace(acc, v)
                else:
                    # generic iterable
                    for i, val in enumerate(v):
                        acc[i] += val
                count += 1

            if count == 0:
                vec = [0.0] * self.dim
                norm = 0.0
            else:
                inv_count = 1.0 / count
                self._vector_scale_inplace(acc, inv_count)
                vec = acc
                norm = self._vec_norm(vec)

                if self.normalize_doc_vectors and norm > 0.0:
                    # normalize in-place to unit vector to speed up cosine (then norm = 1)
                    self._vector_scale_inplace(vec, 1.0 / norm)
                    norm = 1.0

            self.doc_vectors[doc_idx] = vec
            self.doc_norms[doc_idx] = norm

    def _build_query_vector(self, query_terms: Iterable[str]) -> Tuple[List[float], float]:
        """
        Build averaged word2vec vector for the query (same averaging strategy as docs).
        Returns (vector, norm).
        """
        if not query_terms:
            return [0.0] * self.dim, 0.0

        acc = [0.0] * self.dim
        count = 0
        for t in query_terms:
            v = self.embeddings.get(t)
            if v is None:
                continue
            if len(v) != self.dim:
                continue
            # add
            for i, val in enumerate(v):
                acc[i] += val
            count += 1

        if count == 0:
            return [0.0] * self.dim, 0.0

        inv_count = 1.0 / count
        for i in range(self.dim):
            acc[i] *= inv_count

        norm = self._vec_norm(acc)
        if self.normalize_doc_vectors and norm > 0.0:
            # normalize to unit length to match docs
            for i in range(self.dim):
                acc[i] /= norm
            norm = 1.0

        return acc, norm

    def _cosine_similarity(self, q_vec, q_norm, d_vec, d_norm):
        """
        Cosine similarity. If doc/query are normalized to unit length,
        this is simply dot product; else divide by norms.
        """
        if q_norm == 0.0 or d_norm == 0.0:
            return 0.0

        dot = self._vec_dot(q_vec, d_vec)
        if self.normalize_doc_vectors and q_norm == 1.0 and d_norm == 1.0:
            return dot
        return dot / (q_norm * d_norm)

    def rank_documents(self, query_terms: Iterable[str], candidate_doc_indices: Iterable[int]) -> List[Tuple[str, float]]:
        """
        Rank candidate documents (doc indices) by cosine similarity between
        the averaged word2vec query vector and precomputed document vectors.

        Returns:
            list of (pid, score) sorted descending.
        """
        q_vec, q_norm = self._build_query_vector(query_terms)
        if q_norm == 0.0:
            return []

        scores = []
        for doc_idx in candidate_doc_indices:
            d_vec = self.doc_vectors.get(doc_idx)
            d_norm = self.doc_norms.get(doc_idx, 0.0)
            if d_vec is None or d_norm == 0.0:
                score = 0.0
            else:
                score = self._cosine_similarity(q_vec, q_norm, d_vec, d_norm)
            pid = self.pid_list[doc_idx]
            scores.append((pid, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def rank_queries(self, raw_queries: Iterable[str], top_k: int = 20, preprocess=build_query_terms):
        """
        Given an iterable of raw queries (strings), preprocess them using `preprocess`
        (default: build_query_terms) and return top_k results for each query.

        Returns:
            dict: {raw_query: [(pid, score), ...top_k...] }
        """
        results = {}
        for raw_q in raw_queries:
            q_terms = preprocess(raw_q)
            # Use conjunctive search candidate set (same as other rankers)
            candidate_doc_indices = self.index.search_conjunctive(q_terms)
            ranked = self.rank_documents(q_terms, candidate_doc_indices)
            results[raw_q] = ranked[:top_k]
        return results

