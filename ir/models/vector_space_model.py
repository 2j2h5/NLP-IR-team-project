import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from ir.indexing.inverted_index import InvertedIndex
from ir.weighting.tfidf import TFIDFWeighter


class VectorSpaceModel:
    """
    Vector Space Model using TF-IDF weighting and cosine similarity.

    This model:
    - builds sparse TF-IDF document vectors
    - builds sparse TF-IDF query vectors
    - computes cosine similarity between query and documents
    """

    def __init__(
        self,
        index: InvertedIndex,
        tokenizer,
        weighter: TFIDFWeighter,
    ) -> None:
        self.index = index
        self.tokenizer = tokenizer
        self.weighter = weighter

        # Sparse document vectors:
        # doc_id -> {term: tf-idf weight}
        self.doc_vectors: Dict[int, Dict[str, float]] = {}

        # Precomputed L2 norms:
        # doc_id -> ||doc_vector||
        self.doc_norms: Dict[int, float] = {}

    def build(self) -> None:
        """
        Build TF-IDF document vectors and precompute document norms.
        """
        for doc_id in self.index.documents.keys():
            vector = self.weighter.document_vector(doc_id, self.index)
            self.doc_vectors[doc_id] = vector
            self.doc_norms[doc_id] = self._vector_norm(vector)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search documents using cosine similarity.

        Args:
            query: raw query string
            top_k: number of top results to return

        Returns:
            List of (doc_id, score), sorted by descending score
        """
        query_vector = self._build_query_vector(query)
        query_norm = self._vector_norm(query_vector)

        if query_norm == 0.0:
            return []

        scores: Dict[int, float] = defaultdict(float)

        # Dot product: only terms in the query need to be considered
        for term, q_weight in query_vector.items():
            postings = self.index.get_postings(term)

            for doc_id in postings.keys():
                d_weight = self.doc_vectors.get(doc_id, {}).get(term, 0.0)
                if d_weight > 0.0:
                    scores[doc_id] += q_weight * d_weight

        # Normalize by ||q|| * ||d||
        results: List[Tuple[int, float]] = []
        for doc_id, dot_product in scores.items():
            doc_norm = self.doc_norms.get(doc_id, 0.0)

            if doc_norm == 0.0:
                continue

            cosine_score = dot_product / (query_norm * doc_norm)
            results.append((doc_id, cosine_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def explain(
        self,
        query: str,
        doc_id: int,
    ) -> Dict[str, float]:
        """
        Explain per-term contribution to the cosine numerator (dot product).

        Args:
            query: raw query string
            doc_id: target document ID

        Returns:
            {term: contribution}
        """
        if doc_id not in self.doc_vectors:
            raise KeyError(f"Document ID {doc_id} is not built in the model.")

        query_vector = self._build_query_vector(query)
        doc_vector = self.doc_vectors[doc_id]

        contributions: Dict[str, float] = {}

        for term, q_weight in query_vector.items():
            d_weight = doc_vector.get(term, 0.0)
            if d_weight > 0.0:
                contributions[term] = q_weight * d_weight

        return contributions

    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """
        Return the precomputed TF-IDF vector of a document.
        """
        if doc_id not in self.doc_vectors:
            raise KeyError(f"Document ID {doc_id} not found in built vectors.")
        return self.doc_vectors[doc_id]

    def _build_query_vector(self, query: str) -> Dict[str, float]:
        """
        Build a sparse TF-IDF query vector from raw query text.
        """
        query_tokens = self.tokenizer.tokenize(query)
        query_term_counts = Counter(query_tokens)
        return self.weighter.query_vector(query_term_counts, self.index)

    @staticmethod
    def _vector_norm(vector: Dict[str, float]) -> float:
        """
        Compute L2 norm of a sparse vector.
        """
        return math.sqrt(sum(weight * weight for weight in vector.values()))