from __future__ import annotations


def precision_at_k(retrieved, relevant, k: int) -> float:
    if k <= 0:
        return 0.0

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / len(retrieved_k)


def recall_at_k(retrieved, relevant, k: int) -> float:
    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def average_precision(retrieved, relevant) -> float:
    if not relevant:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hit_count += 1
            precision_sum += hit_count / i

    return precision_sum / len(relevant)


def mean_average_precision(ap_list: list[float]) -> float:
    if not ap_list:
        return 0.0
    return sum(ap_list) / len(ap_list)


def f_beta_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive.")

    if precision == 0.0 and recall == 0.0:
        return 0.0

    beta2 = beta * beta
    denominator = beta2 * precision + recall

    if denominator == 0.0:
        return 0.0

    return (1 + beta2) * precision * recall / denominator