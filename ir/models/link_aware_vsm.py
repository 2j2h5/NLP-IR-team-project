from typing import Dict, List, Tuple

from ir.indexing.inverted_index import InvertedIndex
from ir.preprocessors.tokenizer import Tokenizer
from ir.weighting.tfidf import TFIDFWeighter
from ir.models.vector_space_model import VectorSpaceModel
from ir.graph.link_graph import LinkGraph
from ir.graph.pagerank import compute_normalized_pagerank


class LinkAwareVectorSpaceModel:
    """
    Link-aware VSM.

    Final score:
        S(q, d) = alpha * S_text(q, d) + (1 - alpha) * S_link(d)

    S_text:
        TF-IDF cosine similarity from VectorSpaceModel

    S_link:
        normalized in-degree or normalized PageRank
    """

    def __init__(
        self,
        index: InvertedIndex,
        tokenizer: Tokenizer,
        weighter: TFIDFWeighter,
        graph: LinkGraph,
        link_score: str = "pagerank",
        alpha: float = 0.8,
        pagerank_damping: float = 0.85,
        pagerank_max_iter: int = 50,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if link_score not in {"indegree", "pagerank"}:
            raise ValueError("link_score must be either 'indegree' or 'pagerank'.")

        self.index = index
        self.tokenizer = tokenizer
        self.weighter = weighter
        self.graph = graph

        self.link_score = link_score
        self.alpha = alpha
        self.pagerank_damping = pagerank_damping
        self.pagerank_max_iter = pagerank_max_iter

        self.vsm = VectorSpaceModel(
            index=index,
            tokenizer=tokenizer,
            weighter=weighter,
        )

        self.link_scores: Dict[int, float] = {}

    def build(self) -> None:
        self.vsm.build()
        self.link_scores = self._build_link_scores()

    def _build_link_scores(self) -> Dict[int, float]:
        if self.link_score == "indegree":
            scores = self.graph.normalized_indegree_scores()

        elif self.link_score == "pagerank":
            scores = compute_normalized_pagerank(
                graph=self.graph,
                damping=self.pagerank_damping,
                max_iter=self.pagerank_max_iter,
            )

        else:
            raise ValueError(f"Unsupported link_score: {self.link_score}")

        # Ensure every indexed document has a link score.
        return {
            doc_id: scores.get(doc_id, 0.0)
            for doc_id in self.index.documents.keys()
        }

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search documents using interpolated text and link scores.
        """
        text_results = self.vsm.search(
            query=query,
            top_k=len(self.index.documents),
        )

        final_scores: Dict[int, float] = {}

        for doc_id, text_score in text_results:
            link_score = self.link_scores.get(doc_id, 0.0)

            final_score = (
                self.alpha * text_score
                + (1.0 - self.alpha) * link_score
            )

            final_scores[doc_id] = final_score

        ranked_results = sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked_results[:top_k]

    def explain(
        self,
        query: str,
        doc_id: int,
    ) -> Dict[str, float]:
        """
        Return score components for debugging or result explanation.
        """
        text_score = dict(
            self.vsm.search(query=query, top_k=len(self.index.documents))
        ).get(doc_id, 0.0)

        link_score = self.link_scores.get(doc_id, 0.0)

        final_score = (
            self.alpha * text_score
            + (1.0 - self.alpha) * link_score
        )

        return {
            "text_score": text_score,
            "link_score": link_score,
            "alpha": self.alpha,
            "final_score": final_score,
        }