"""
Evaluation metrics for search engine performance assessment.
Implements all metrics required for Part 2: Evaluation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


class SearchEvaluator:
    """
    Evaluator class for search engine metrics.

    Implements:
    - Precision@K (P@K)
    - Recall@K (R@K)
    - Average Precision@K (AP@K)
    - F1-Score@K
    - Mean Average Precision (MAP)
    - Mean Reciprocal Rank (MRR)
    - Normalized Discounted Cumulative Gain (NDCG)
    """

    def __init__(self, ground_truth: Dict[str, int]):
        """
        Initialize evaluator with ground truth relevance labels.

        Args:
            ground_truth: Dict {pid: relevance_label} where label is 0 or 1
        """
        self.ground_truth = ground_truth
        self.total_relevant = sum(1 for label in ground_truth.values() if label == 1)

    def precision_at_k(self, ranked_pids: List[str], k: int) -> float:
        """Precision@K: Proportion of relevant documents in top-K results."""
        if k == 0 or not ranked_pids:
            return 0.0

        top_k = ranked_pids[:k]
        relevant_in_top_k = sum(1 for pid in top_k if self.ground_truth.get(pid, 0) == 1)

        return relevant_in_top_k / k

    def recall_at_k(self, ranked_pids: List[str], k: int) -> float:
        """Recall@K: Proportion of all relevant documents retrieved in top-K."""
        if self.total_relevant == 0 or not ranked_pids:
            return 0.0

        top_k = ranked_pids[:k]
        relevant_in_top_k = sum(1 for pid in top_k if self.ground_truth.get(pid, 0) == 1)

        return relevant_in_top_k / self.total_relevant

    def average_precision_at_k(self, ranked_pids: List[str], k: int) -> float:
        """Average Precision@K: Average of precision values at each relevant document position."""
        if self.total_relevant == 0 or not ranked_pids:
            return 0.0

        top_k = ranked_pids[:k]
        precision_sum = 0.0
        relevant_count = 0

        for i, pid in enumerate(top_k, start=1):
            if self.ground_truth.get(pid, 0) == 1:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        normalizer = min(k, self.total_relevant)
        return precision_sum / normalizer if normalizer > 0 else 0.0

    def f1_score_at_k(self, ranked_pids: List[str], k: int) -> float:
        """F1-Score@K: Harmonic mean of Precision@K and Recall@K."""
        precision = self.precision_at_k(ranked_pids, k)
        recall = self.recall_at_k(ranked_pids, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def reciprocal_rank(self, ranked_pids: List[str]) -> float:
        """Reciprocal Rank: 1 / rank of first relevant document."""
        for i, pid in enumerate(ranked_pids, start=1):
            if self.ground_truth.get(pid, 0) == 1:
                return 1.0 / i
        return 0.0

    def dcg_at_k(self, ranked_pids: List[str], k: int) -> float:
        """Discounted Cumulative Gain@K."""
        dcg = 0.0
        top_k = ranked_pids[:k]

        for i, pid in enumerate(top_k, start=1):
            relevance = self.ground_truth.get(pid, 0)
            dcg += relevance / np.log2(i + 1)

        return dcg

    def ndcg_at_k(self, ranked_pids: List[str], k: int) -> float:
        """Normalized Discounted Cumulative Gain@K."""
        dcg = self.dcg_at_k(ranked_pids, k)

        ideal_ranking = sorted(
            self.ground_truth.items(),
            key=lambda x: x[1],
            reverse=True
        )
        ideal_pids = [pid for pid, _ in ideal_ranking]
        idcg = self.dcg_at_k(ideal_pids, k)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate_all(self, ranked_pids: List[str], k_values: List[int] = [5, 10, 20]) -> Dict:
        """Evaluate all metrics at multiple K values."""
        results = {
            'MRR': self.reciprocal_rank(ranked_pids)
        }

        for k in k_values:
            results[f'P@{k}'] = self.precision_at_k(ranked_pids, k)
            results[f'R@{k}'] = self.recall_at_k(ranked_pids, k)
            results[f'AP@{k}'] = self.average_precision_at_k(ranked_pids, k)
            results[f'F1@{k}'] = self.f1_score_at_k(ranked_pids, k)
            results[f'NDCG@{k}'] = self.ndcg_at_k(ranked_pids, k)

        return results


def load_validation_labels(csv_path: str) -> Dict[int, Dict[str, int]]:
    """Load ground truth labels from validation_labels.csv."""
    df = pd.read_csv(csv_path)

    labels_by_query = {}
    for query_id in df['query_id'].unique():
        query_data = df[df['query_id'] == query_id]
        labels_by_query[query_id] = dict(zip(query_data['pid'], query_data['labels']))

    return labels_by_query


def calculate_map(evaluators: List[SearchEvaluator],
                  all_ranked_results: List[List[str]],
                  k: int = 20) -> float:
    """Mean Average Precision: Average of AP@K across all queries."""
    if not evaluators or not all_ranked_results:
        return 0.0

    ap_sum = sum(
        evaluator.average_precision_at_k(ranked_pids, k)
        for evaluator, ranked_pids in zip(evaluators, all_ranked_results)
    )

    return ap_sum / len(evaluators)


def calculate_mrr(evaluators: List[SearchEvaluator],
                  all_ranked_results: List[List[str]]) -> float:
    """Mean Reciprocal Rank: Average of RR across all queries."""
    if not evaluators or not all_ranked_results:
        return 0.0

    rr_sum = sum(
        evaluator.reciprocal_rank(ranked_pids)
        for evaluator, ranked_pids in zip(evaluators, all_ranked_results)
    )

    return rr_sum / len(evaluators)
