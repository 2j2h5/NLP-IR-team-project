from typing import List, Set


def precision_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Precision@k.

    Precision@k = (# of relevant documents in top-k) / k

    Args:
        retrieved: ranked list of retrieved document IDs
        relevant: set of relevant document IDs
        k: cutoff rank

    Returns:
        Precision@k score
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0

    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_count / k


def recall_at_k(
    retrieved: List[int],
    relevant: Set[int],
    k: int,
) -> float:
    """
    Compute Recall@k.

    Recall@k = (# of relevant documents in top-k) / (# of all relevant documents)

    Args:
        retrieved: ranked list of retrieved document IDs
        relevant: set of relevant document IDs
        k: cutoff rank

    Returns:
        Recall@k score
    """
    if k <= 0:
        raise ValueError("k must be positive.")

    if not relevant:
        return 0.0

    retrieved_k = retrieved[:k]
    relevant_count = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_count / len(relevant)


def average_precision(
    retrieved: List[int],
    relevant: Set[int],
) -> float:
    """
    Compute Average Precision (AP).

    AP is the average of precision values at the ranks
    where relevant documents are retrieved.

    Args:
        retrieved: ranked list of retrieved document IDs
        relevant: set of relevant document IDs

    Returns:
        Average Precision score
    """
    if not relevant:
        return 0.0

    hit_count = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hit_count += 1
            precision_sum += hit_count / rank

    return precision_sum / len(relevant)


def mean_average_precision(
    all_retrieved: List[List[int]],
    all_relevant: List[Set[int]],
) -> float:
    """
    Compute Mean Average Precision (MAP).

    Args:
        all_retrieved: list of ranked retrieval results for each query
        all_relevant: list of relevant document ID sets for each query

    Returns:
        MAP score
    """
    if len(all_retrieved) != len(all_relevant):
        raise ValueError("all_retrieved and all_relevant must have the same length.")

    if not all_retrieved:
        return 0.0

    ap_sum = 0.0
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        ap_sum += average_precision(retrieved, relevant)

    return ap_sum / len(all_retrieved)