import random
import numpy as np

from myapp.search.objects import Document

from myapp.search.algorithms import InvertedIndex, TFIDFRanker, BM25Ranker, CustomScoreRanker
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

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        results = dummy_search(corpus, search_id)  # replace with call to search algorithm

        # results = search_in_corpus(search_query)
        return results

    def search_tfidf(self,query, corpus, top_k: int):
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

        # Rank candidates using TF-IDF + cosine similarity
        ranker = TFIDFRanker(inv_index)
        ranked_results = ranker.rank_documents(query_terms, candidate_doc_indices)

        # Return top K results
        return ranked_results[:top_k]

    def search_bm25(self, query, corpus, top_k: int):
        """
        Search documents using conjunctive query (AND) and BM25 ranking.
        """
        query_terms = build_query_terms(query)
        print(f"Processed query terms (BM25): {query_terms}")

        if not query_terms:
            return []

        inv_index = InvertedIndex(corpus)
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        if not candidate_doc_indices:
            print("No documents found matching all query terms (BM25)")
            return []

        print(f"Found {len(candidate_doc_indices)} documents matching all terms (BM25)")

        ranker = BM25Ranker(inv_index)
        ranked_results = ranker.rank_documents(query_terms, candidate_doc_indices)
        return ranked_results[:top_k]

    def search_custom(self, query, corpus, top_k: int, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1, delta: float = 0.5):
        """
        Search documents using conjunctive query (AND) and a custom ranking
        that combines BM25 with rating, discount, and out-of-stock signals.

        Args:
            query: raw query string
            corpus: preprocessed corpus dict {pid: preprocessed_doc}
            top_k: number of results to return
            alpha: weight for BM25 score
            beta: weight for normalized rating
            gamma: weight for normalized discount
            delta: penalty weight for out-of-stock documents

        Returns:
            list of (pid, score) tuples ranked by the custom score
        """
        # 1. Preprocess query
        query_terms = build_query_terms(query)
        print(f"Processed query terms (Custom): {query_terms}")

        if not query_terms:
            return []

        # 2. Build inverted index and candidate set (AND semantics)
        inv_index = InvertedIndex(corpus)
        candidate_doc_indices = inv_index.search_conjunctive(query_terms)

        if not candidate_doc_indices:
            print("No documents found matching all query terms (Custom)")
            return []

        print(f"Found {len(candidate_doc_indices)} documents matching all terms (Custom)")

        # 3. Base BM25 ranker
        bm25_ranker = BM25Ranker(inv_index)

        # 4. Custom ranker on top of BM25 + numerical signals
        custom_ranker = CustomScoreRanker(base_ranker=bm25_ranker, corpus=corpus, alpha=alpha, beta=beta, gamma=gamma, delta=delta)
        ranked_results = custom_ranker.rank_documents(query_terms, candidate_doc_indices)
        return ranked_results[:top_k]
