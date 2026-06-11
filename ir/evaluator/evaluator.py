from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

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
        self.model = model

    def evaluate_query(
        self,
        query: str,
        relevant_docs: Set[int],
        k: int = 10,
    ) -> Dict[str, float]:
        results = self.model.search(query, top_k=k)
        retrieved_ids = [doc_id for doc_id, _ in results]

        return self._compute_query_metrics(
            retrieved_ids=retrieved_ids,
            relevant_docs=relevant_docs,
            k=k,
        )

    def evaluate_query_with_previous_query(
        self,
        query: str,
        relevant_docs: Set[int],
        previous_query: Optional[str] = None,
        k: int = 10,
    ) -> Dict[str, float]:
        results = self.model.search(
            query=query,
            previous_query=previous_query,
            top_k=k,
        )
        retrieved_ids = [doc_id for doc_id, _ in results]

        return self._compute_query_metrics(
            retrieved_ids=retrieved_ids,
            relevant_docs=relevant_docs,
            k=k,
        )

    def evaluate_all(
        self,
        queries: Dict[int, str],
        relevance_judgments: Dict[int, Set[int]],
        k: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        return self._evaluate(
            queries=queries,
            relevance_judgments=relevance_judgments,
            k=k,
            verbose=verbose,
            use_previous_query=False,
        )

    def evaluate_all_with_previous_queries(
        self,
        queries: Dict[int, str],
        relevance_judgments: Dict[int, Set[int]],
        k: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        return self._evaluate(
            queries=queries,
            relevance_judgments=relevance_judgments,
            k=k,
            verbose=verbose,
            use_previous_query=True,
        )

    def _evaluate(
        self,
        queries: Dict[int, str],
        relevance_judgments: Dict[int, Set[int]],
        k: int,
        verbose: bool,
        use_previous_query: bool,
    ) -> Dict[str, Any]:
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

        for query_id in sorted(queries.keys()):
            if query_id not in relevance_judgments:
                continue

            query_text = queries[query_id]
            relevant_docs = relevance_judgments[query_id]

            previous_query = None
            if use_previous_query:
                step = query_id % 10
                if step > 1:
                    previous_query = queries.get(query_id - 1)

                query_result = self.evaluate_query_with_previous_query(
                    query=query_text,
                    previous_query=previous_query,
                    relevant_docs=relevant_docs,
                    k=k,
                )
            else:
                query_result = self.evaluate_query(
                    query=query_text,
                    relevant_docs=relevant_docs,
                    k=k,
                )

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
                self._print_query_result(
                    query_id=query_id,
                    query_result=query_result,
                    k=k,
                    previous_query_id=(
                        query_id - 1 if use_previous_query and query_id % 10 > 1 else None
                    ),
                )

        return self._build_summary(
            per_query_results=per_query_results,
            ap_list=ap_list,
            precision_sum=precision_sum,
            recall_sum=recall_sum,
            f025_sum=f025_sum,
            f05_sum=f05_sum,
            f1_sum=f1_sum,
            f2_sum=f2_sum,
            f4_sum=f4_sum,
            evaluated_query_count=evaluated_query_count,
            k=k,
        )

    def _compute_query_metrics(
        self,
        retrieved_ids: List[int],
        relevant_docs: Set[int],
        k: int,
    ) -> Dict[str, float]:
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

    def _build_summary(
        self,
        per_query_results: Dict[int, Dict[str, float]],
        ap_list: List[float],
        precision_sum: float,
        recall_sum: float,
        f025_sum: float,
        f05_sum: float,
        f1_sum: float,
        f2_sum: float,
        f4_sum: float,
        evaluated_query_count: int,
        k: int,
    ) -> Dict[str, Any]:
        if evaluated_query_count == 0:
            return {
                "per_query": per_query_results,
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

        return {
            "per_query": per_query_results,
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

    @staticmethod
    def _print_query_result(
        query_id: int,
        query_result: Dict[str, float],
        k: int,
        previous_query_id: Optional[int] = None,
    ) -> None:
        if previous_query_id is None:
            print(f"[Query {query_id}]")
        else:
            print(f"[Query {query_id} | previous={previous_query_id}]")

        print(f"  Precision@{k}: {query_result[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:    {query_result[f'recall@{k}']:.4f}")
        print(f"  F0.25@{k}:     {query_result[f'f0.25@{k}']:.4f}")
        print(f"  F0.5@{k}:      {query_result[f'f0.5@{k}']:.4f}")
        print(f"  F1@{k}:        {query_result[f'f1@{k}']:.4f}")
        print(f"  F2@{k}:        {query_result[f'f2@{k}']:.4f}")
        print(f"  F4@{k}:        {query_result[f'f4@{k}']:.4f}")
        print(f"  AP:            {query_result['average_precision']:.4f}")
        print()