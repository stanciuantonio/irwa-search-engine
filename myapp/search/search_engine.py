import os
import random
import numpy as np
from typing import Optional, List

from myapp.search.objects import Document
from myapp.search.algorithms import InvertedIndex, TFIDFRanker, BM25Ranker, CustomScoreRanker, Word2VecRanker
from myapp.preprocessing.text_processing import build_query_terms


def dummy_search(corpus: dict, search_id, num_results=20):
    """
    Just a demo method, that returns random <num_results> documents from the corpus
    :param corpus: the documents corpus
    :param search_id: the search id
    :param num_results: number of documents to return
    :return: a list of random documents from the corpus
    """
    res = []
    doc_ids = list(corpus.keys())
    docs_to_return = np.random.choice(doc_ids, size=num_results, replace=False)
    for doc_id in docs_to_return:
        doc = corpus[doc_id]
        res.append(Document(pid=doc.pid, title=doc.title, description=doc.description,
                            url="doc_details?pid={}&search_id={}&param2=2".format(doc.pid, search_id), ranking=random.random()))
    return res


class SearchEngine:
    """Class that implements the search engine logic"""

    def __init__(self):
        self.w2v_embeddings = None
        self._load_word2vec_embeddings()

    def _load_word2vec_embeddings(self):
        """Load Word2Vec embeddings if available"""
        try:
            import gensim.models
            embeddings_path = os.getenv("WORD2VEC_MODEL_PATH", "data/word2vec_model.bin")
            if os.path.exists(embeddings_path):
                model = gensim.models.KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
                self.w2v_embeddings = model.wv
                print("[SearchEngine] Word2Vec embeddings loaded successfully")
        except Exception as e:
            print(f"[SearchEngine] Warning: Could not load Word2Vec embeddings: {e}")
            self.w2v_embeddings = None

    def search(self, search_query: str, search_id: int, corpus: dict,
               algorithm: str = "tfidf", top_k: int = 20) -> List[Document]:
        """
        Search documents using conjunctive query (AND) and TF-IDF ranking

        This function implements the complete pipeline:
        1. Preprocess query
        2. Build inverted index
        3. Find documents with ALL query terms (conjunctive/AND)
        4. Rank results by TF-IDF

        Args:
            search_query: user's search query string
            search_id: unique identifier for this search (from analytics)
            corpus: preprocessed document corpus
            algorithm: "tfidf", "bm25", "custom", or "word2vec"
            top_k: number of results to return

        Returns:
            List of Document objects with ranking scores
        """
        print(f"[SearchEngine] Executing search: '{search_query}' with algorithm={algorithm}")

        # Validate algorithm
        valid_algorithms = ["tfidf", "bm25", "custom", "word2vec"]
        if algorithm not in valid_algorithms:
            print(f"[SearchEngine] Warning: Unknown algorithm '{algorithm}', using tfidf")
            algorithm = "tfidf"

        # Route to appropriate search method
        if algorithm == "tfidf":
            ranked_results = self.search_tfidf(search_query, corpus, top_k)
        elif algorithm == "bm25":
            ranked_results = self.search_bm25(search_query, corpus, top_k)
        elif algorithm == "custom":
            ranked_results = self.search_custom(search_query, corpus, top_k)
        elif algorithm == "word2vec":
            ranked_results = self.search_word2vec(search_query, corpus, top_k)
        else:
            ranked_results = []

        # Convert (pid, score) tuples to Document objects
        results_docs = []
        for position, (pid, score) in enumerate(ranked_results, 1):
            if pid in corpus:
                # Get the original document data
                original_data = corpus[pid].get("original", {})

                # Create Document object with ranking info
                doc = Document(
                    pid=pid,
                    title=original_data.get("title", ""),
                    description=original_data.get("description", ""),
                    brand=original_data.get("brand"),
                    category=original_data.get("category"),
                    sub_category=original_data.get("sub_category"),
                    product_details=original_data.get("product_details"),
                    seller=original_data.get("seller"),
                    out_of_stock=original_data.get("out_of_stock", False),
                    selling_price=original_data.get("selling_price"),
                    discount=original_data.get("discount"),
                    actual_price=original_data.get("actual_price"),
                    average_rating=original_data.get("average_rating"),
                    url=f"doc_details?pid={pid}&search_id={search_id}",
                    images=original_data.get("images"),
                    ranking_score=round(score, 4),
                    algorithm_used=algorithm,
                    ranking_position=position
                )
                results_docs.append(doc)

        print(f"[SearchEngine] Found {len(results_docs)} results")
        return results_docs

    def search_tfidf(self, query: str, corpus: dict, top_k: int) -> List[tuple]:
        """Search with TF-IDF + cosine similarity ranking"""
        query_terms = build_query_terms(query)
        print(f"Processed query terms: {query_terms}")

        if not query_terms:
            return []

        # 1. Build inverted index
        inv_index = InvertedIndex(corpus)

        # 2. Find documents containing ALL query terms (AND/conjunctive)
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        # If no documents match all terms, return empty
        if not candidate_doc_indices:
            print("[TF-IDF] No documents found matching all query terms")
            return []

        print(f"[TF-IDF] Found {len(candidate_doc_indices)} documents matching all terms")

        # 3. Rank candidates using TF-IDF + cosine similarity
        ranker = TFIDFRanker(inv_index)
        ranked_results = ranker.rank_documents(query_terms, candidate_doc_indices)

        # 4. Return top K results
        return ranked_results[:top_k]

    def search_bm25(self, query: str, corpus: dict, top_k: int) -> List[tuple]:
        """Search with BM25 ranking"""
        query_terms = build_query_terms(query)
        print(f"[BM25] Processed query terms: {query_terms}")

        if not query_terms:
            return []

        # 1. Build inverted index
        inv_index = InvertedIndex(corpus)
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        if not candidate_doc_indices:
            print("[BM25] No documents found matching all query terms")
            return []

        print(f"[BM25] Found {len(candidate_doc_indices)} documents matching all terms")

        # 2. Rank candidates base on BM25
        ranker = BM25Ranker(inv_index)
        ranked_results = ranker.rank_documents(query_terms, candidate_doc_indices)

        # 3. Return top k results
        return ranked_results[:top_k]

    def search_custom(self, query: str, corpus: dict, top_k: int,
                     alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1, delta: float = 0.5) -> List[tuple]:
        """Search with custom ranking (BM25 + rating + discount + stock)"""
        query_terms = build_query_terms(query)
        print(f"[Custom] Processed query terms: {query_terms}")

        if not query_terms:
            return []

        # 1. Build inverted index and candidate set (AND semantics)
        inv_index = InvertedIndex(corpus)
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        if not candidate_doc_indices:
            print("[Custom] No documents found matching all query terms")
            return []

        print(f"[Custom] Found {len(candidate_doc_indices)} documents matching all terms")

        # 2. Base BM25 ranker
        bm25_ranker = BM25Ranker(inv_index)

        # 3. Custom ranker on top of BM25 + numerical signals
        custom_ranker = CustomScoreRanker(base_ranker=bm25_ranker, corpus=corpus, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        ranked_results = custom_ranker.rank_documents(query_terms, candidate_doc_indices)

        return ranked_results[:top_k]

    def search_word2vec(self, query: str, corpus: dict, top_k: int) -> List[tuple]:
        """Search with Word2Vec + cosine similarity ranking"""

        if self.w2v_embeddings is None:
            print("[Word2Vec] Embeddings not available, falling back to TF-IDF")
            return self.search_tfidf(query, corpus, top_k)

        # 1. Preprocess query
        query_terms = build_query_terms(query)
        print(f"[Word2Vec] Processed query terms: {query_terms}")

        if not query_terms:
            return []

        # 2. Build inverted index
        inv_index = InvertedIndex(corpus)

        # 3. Conjunctive (AND) candidate lookup
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        if not candidate_doc_indices:
            print("[Word2Vec] No documents found matching all query terms")
            return []

        print(f"[Word2Vec] Found {len(candidate_doc_indices)} documents matching all terms")

        w2v_ranker = Word2VecRanker(inv_index, self.w2v_embeddings)
        ranked_results = w2v_ranker.rank_documents(query_terms, candidate_doc_indices)

        # 4. Return top-K
        return ranked_results[:top_k]
