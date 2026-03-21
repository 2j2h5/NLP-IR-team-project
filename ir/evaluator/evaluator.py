from typing import Dict, List, Set, Any

from ir.evaluator.metrics import (
    precision_at_k,
    recall_at_k,
    average_precision,
    mean_average_precision,
)


class Evaluator:
    """
    Evaluator for IR retrieval models.

    This class assumes the model has a method:
        search(query: str, top_k: int) -> List[Tuple[int, float]]

    where each result is:
        (doc_id, score)
    """

    def __init__(self, model: Any) -> None:
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
            query: raw query string
            relevant_docs: set of relevant document IDs for the query
            k: cutoff rank

        Returns:
            Dictionary containing P@k, R@k, and AP
        """
        results = self.model.search(query, top_k=k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        return {
            f"precision@{k}": precision_at_k(retrieved_ids, relevant_docs, k),
            f"recall@{k}": recall_at_k(retrieved_ids, relevant_docs, k),
            "average_precision": average_precision(retrieved_ids, relevant_docs),
        }

    def evaluate_all(
        self,
        queries: Dict[int, str],
        relevance: Dict[int, Set[int]],
        k: int = 10,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate all queries and compute MAP.

        Args:
            queries: {query_id: query_text}
            relevance: {query_id: set_of_relevant_doc_ids}
            k: cutoff rank
            verbose: whether to print per-query results

        Returns:
            Dictionary with:
                - per_query: {query_id: metric dict}
                - map: MAP score
        """
        per_query_results: Dict[int, Dict[str, float]] = {}
        all_retrieved: List[List[int]] = []
        all_relevant: List[Set[int]] = []

        for query_id, query_text in queries.items():
            results = self.model.search(query_text, top_k=k)
            retrieved_ids = [doc_id for doc_id, _ in results]
            relevant_docs = relevance.get(query_id, set())

            query_metrics = {
                f"precision@{k}": precision_at_k(retrieved_ids, relevant_docs, k),
                f"recall@{k}": recall_at_k(retrieved_ids, relevant_docs, k),
                "average_precision": average_precision(retrieved_ids, relevant_docs),
            }

            per_query_results[query_id] = query_metrics
            all_retrieved.append(retrieved_ids)
            all_relevant.append(relevant_docs)

            if verbose:
                print(
                    f"[Query {query_id}] "
                    f"P@{k}: {query_metrics[f'precision@{k}']:.4f}, "
                    f"R@{k}: {query_metrics[f'recall@{k}']:.4f}, "
                    f"AP: {query_metrics['average_precision']:.4f}"
                )

        map_score = mean_average_precision(all_retrieved, all_relevant)

        if verbose:
            print(f"\nMAP: {map_score:.4f}")

        return {
            "per_query": per_query_results,
            "map": map_score,
        }