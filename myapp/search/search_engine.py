import random
import numpy as np

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

    def search(self, search_query, search_id, corpus):
        print("Search query:", search_query)

        results = []
        ### You should implement your search logic here:
        results = dummy_search(corpus, search_id)  # replace with call to search algorithm

        # results = search_in_corpus(search_query)
        return results

    def search_tfidf(self,query, corpus, top_k=20):
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
