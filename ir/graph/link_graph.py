from collections import defaultdict
from typing import Dict, Iterable, List, Set, Tuple


class LinkGraph:
    """
    Directed document link graph.

    Edge direction:
        source_doc_id -> target_doc_id
    """

    def __init__(self, edges: Iterable[Tuple[int, int]] = None) -> None:
        self.out_neighbors: Dict[int, Set[int]] = defaultdict(set)
        self.in_neighbors: Dict[int, Set[int]] = defaultdict(set)
        self.nodes: Set[int] = set()

        if edges is not None:
            self.add_edges(edges)

    def add_edge(self, source: int, target: int) -> None:
        if source == target:
            return

        self.out_neighbors[source].add(target)
        self.in_neighbors[target].add(source)

        self.nodes.add(source)
        self.nodes.add(target)

    def add_edges(self, edges: Iterable[Tuple[int, int]]) -> None:
        for source, target in edges:
            self.add_edge(int(source), int(target))

    def get_nodes(self) -> Set[int]:
        return set(self.nodes)

    def get_out_neighbors(self, doc_id: int) -> Set[int]:
        return self.out_neighbors.get(doc_id, set())

    def get_in_neighbors(self, doc_id: int) -> Set[int]:
        return self.in_neighbors.get(doc_id, set())

    def out_degree(self, doc_id: int) -> int:
        return len(self.get_out_neighbors(doc_id))

    def in_degree(self, doc_id: int) -> int:
        return len(self.get_in_neighbors(doc_id))

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return sum(len(targets) for targets in self.out_neighbors.values())

    def normalized_indegree_scores(self) -> Dict[int, float]:
        """
        Return min-max normalized in-degree scores in [0, 1].
        """
        if not self.nodes:
            return {}

        raw_scores = {
            doc_id: float(self.in_degree(doc_id))
            for doc_id in self.nodes
        }

        max_score = max(raw_scores.values())
        min_score = min(raw_scores.values())

        if max_score == min_score:
            return {
                doc_id: 0.0
                for doc_id in self.nodes
            }

        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in raw_scores.items()
        }

    @classmethod
    def from_edges(cls, edges: List[Tuple[int, int]]) -> "LinkGraph":
        return cls(edges)