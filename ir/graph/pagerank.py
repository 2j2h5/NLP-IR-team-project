from typing import Dict

from ir.graph.link_graph import LinkGraph


def compute_pagerank(
    graph: LinkGraph,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[int, float]:
    """
    Compute PageRank scores for a directed graph.

    PR(d) = (1 - damping) / N
            + damping * sum(PR(u) / out_degree(u))
    """
    nodes = sorted(graph.get_nodes())
    n = len(nodes)

    if n == 0:
        return {}

    initial_score = 1.0 / n
    scores = {
        node: initial_score
        for node in nodes
    }

    for _ in range(max_iter):
        new_scores = {
            node: (1.0 - damping) / n
            for node in nodes
        }

        dangling_mass = 0.0

        for node in nodes:
            out_neighbors = graph.get_out_neighbors(node)

            if not out_neighbors:
                dangling_mass += scores[node]
                continue

            share = scores[node] / len(out_neighbors)

            for target in out_neighbors:
                if target in new_scores:
                    new_scores[target] += damping * share

        dangling_share = damping * dangling_mass / n

        for node in nodes:
            new_scores[node] += dangling_share

        diff = sum(
            abs(new_scores[node] - scores[node])
            for node in nodes
        )

        scores = new_scores

        if diff < tol:
            break

    return scores


def normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
    """
    Min-max normalize scores to [0, 1].
    """
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())

    if max_score == min_score:
        return {
            doc_id: 0.0
            for doc_id in scores
        }

    return {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores.items()
    }


def compute_normalized_pagerank(
    graph: LinkGraph,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[int, float]:
    raw_scores = compute_pagerank(
        graph=graph,
        damping=damping,
        max_iter=max_iter,
        tol=tol,
    )

    return normalize_scores(raw_scores)