from __future__ import annotations

from typing import Dict, List, Set, Any

from ir.evaluator.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    mean_average_precision,
    f_beta_score,
)


class Evaluator:
    """
    Evaluate retrieval models using Precision, Recall, F-beta, and MAP.
    """

    def __init__(self, model: Any):
        """
        Args:
            model: Retrieval model that provides a search(query, top_k) method.
        """
        self.model = model

    def evaluate_query(
        self,
        query: str,
        relevant_docs: Set[int],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate a single query.

        Args:
            query: Query string
            relevant_docs: Set of relevant document IDs
            k: Cutoff for top-k evaluation

        Returns:
            Dictionary containing evaluation metrics for the query
        """
        results = self.model.search(query, top_k=k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        p_at_k = precision_at_k(retrieved_ids, relevant_docs, k)
        r_at_k = recall_at_k(retrieved_ids, relevant_docs, k)

        return {
            f"precision@{k}": p_at_k,
            f"recall@{k}": r_at_k,
            f"f0.25@{k}": f_beta_score(p_at_k, r_at_k, beta=0.25),
            f"f0.5@{k}": f_beta_score(p_at_k, r_at_k, beta=0.5),
            f"f1@{k}": f_beta_score(p_at_k, r_at_k, beta=1.0),
            f"f2@{k}": f_beta_score(p_at_k, r_at_k, beta=2.0),
            f"f4@{k}": f_beta_score(p_at_k, r_at_k, beta=4.0),
            "average_precision": average_precision(retrieved_ids, relevant_docs),
        }

    def evaluate_all(
        self,
        queries: Dict[int, str],
        relevance_judgments: Dict[int, Set[int]],
        k: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate all queries and compute mean metrics.

        Args:
            queries: Mapping from query_id to query text
            relevance_judgments: Mapping from query_id to relevant doc ID set
            k: Cutoff for top-k evaluation
            verbose: Whether to print per-query results

        Returns:
            Dictionary containing per-query results and overall summary
        """
        per_query_results: Dict[int, Dict[str, float]] = {}
        ap_list: List[float] = []

        precision_sum = 0.0
        recall_sum = 0.0
        f025_sum = 0.0
        f05_sum = 0.0
        f1_sum = 0.0
        f2_sum = 0.0
        f4_sum = 0.0

        evaluated_query_count = 0

        for query_id, query_text in queries.items():
            if query_id not in relevance_judgments:
                continue

            relevant_docs = relevance_judgments[query_id]
            query_result = self.evaluate_query(query_text, relevant_docs, k=k)

            per_query_results[query_id] = query_result
            ap_list.append(query_result["average_precision"])

            precision_sum += query_result[f"precision@{k}"]
            recall_sum += query_result[f"recall@{k}"]
            f025_sum += query_result[f"f0.25@{k}"]
            f05_sum += query_result[f"f0.5@{k}"]
            f1_sum += query_result[f"f1@{k}"]
            f2_sum += query_result[f"f2@{k}"]
            f4_sum += query_result[f"f4@{k}"]

            evaluated_query_count += 1

            if verbose:
                print(f"[Query {query_id}]")
                print(f"  Precision@{k}: {query_result[f'precision@{k}']:.4f}")
                print(f"  Recall@{k}:    {query_result[f'recall@{k}']:.4f}")
                print(f"  F0.25@{k}:     {query_result[f'f0.25@{k}']:.4f}")
                print(f"  F0.5@{k}:      {query_result[f'f0.5@{k}']:.4f}")
                print(f"  F1@{k}:        {query_result[f'f1@{k}']:.4f}")
                print(f"  F2@{k}:        {query_result[f'f2@{k}']:.4f}")
                print(f"  F4@{k}:        {query_result[f'f4@{k}']:.4f}")
                print(f"  AP:           {query_result['average_precision']:.4f}")
                print()

        if evaluated_query_count == 0:
            summary = {
                f"mean_precision@{k}": 0.0,
                f"mean_recall@{k}": 0.0,
                f"mean_f0.25@{k}": 0.0,
                f"mean_f0.5@{k}": 0.0,
                f"mean_f1@{k}": 0.0,
                f"mean_f2@{k}": 0.0,
                f"mean_f4@{k}": 0.0,
                "map": 0.0,
                "num_queries": 0,
            }
        else:
            summary = {
                f"mean_precision@{k}": precision_sum / evaluated_query_count,
                f"mean_recall@{k}": recall_sum / evaluated_query_count,
                f"mean_f0.25@{k}": f025_sum / evaluated_query_count,
                f"mean_f0.5@{k}": f05_sum / evaluated_query_count,
                f"mean_f1@{k}": f1_sum / evaluated_query_count,
                f"mean_f2@{k}": f2_sum / evaluated_query_count,
                f"mean_f4@{k}": f4_sum / evaluated_query_count,
                "map": mean_average_precision(ap_list),
                "num_queries": evaluated_query_count,
            }

        return {
            "per_query": per_query_results,
            **summary,
        }