import math
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ir.indexing.inverted_index import InvertedIndex
from ir.models.vector_space_model import VectorSpaceModel
from ir.weighting.tfidf import TFIDFWeighter


class IntentionAwareVectorSpaceModel:
    def __init__(
        self,
        index: InvertedIndex,
        tokenizer,
        weighter: TFIDFWeighter,
        paragraph_embeddings: Dict[int, np.ndarray],
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        alpha: float = 0.8,
        intent_k: int = 3,
        temperature: float = 0.1,
        normalize_scores: bool = True,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")
        if intent_k <= 0:
            raise ValueError("intent_k must be positive.")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

        self.index = index
        self.tokenizer = tokenizer
        self.weighter = weighter
        self.paragraph_embeddings = paragraph_embeddings
        self.embedding_model_name = embedding_model_name

        self.alpha = alpha
        self.intent_k = intent_k
        self.temperature = temperature
        self.normalize_scores = normalize_scores

        self.vsm = VectorSpaceModel(
            index=index,
            tokenizer=tokenizer,
            weighter=weighter,
        )

        self.embedding_model = SentenceTransformer(embedding_model_name)

        self.doc_intent_vectors: Dict[int, np.ndarray] = {}

    def build(self) -> None:
        self.vsm.build()
        self.doc_intent_vectors = self._build_document_intent_vectors()

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        text_results = self.vsm.search(
            query=query,
            top_k=len(self.index.documents),
        )

        text_scores = {
            doc_id: score
            for doc_id, score in text_results
        }

        query_intent = self._build_query_intent_vector(query)

        intent_scores = {
            doc_id: self._intention_score(query_intent, doc_id)
            for doc_id in self.index.documents.keys()
        }

        if self.normalize_scores:
            text_scores = self._minmax_normalize(text_scores)
            intent_scores = self._minmax_normalize(intent_scores)

        final_scores: Dict[int, float] = {}

        for doc_id in self.index.documents.keys():
            text_score = text_scores.get(doc_id, 0.0)
            intent_score = intent_scores.get(doc_id, 0.0)

            final_scores[doc_id] = (
                self.alpha * text_score
                + (1.0 - self.alpha) * intent_score
            )

        ranked_results = sorted(
            final_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked_results[:top_k]

    def _build_query_intent_vector(self, query: str) -> np.ndarray:
        embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embedding = np.asarray(embedding, dtype=np.float32)

        # 현재 단일 쿼리 실험:
        # q_intent = normalize(q - 0) = normalize(q)
        return self._normalize_dense_or_zero(embedding)

    def _build_document_intent_vectors(self) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}

        for doc_id in self.index.documents.keys():
            paragraph_vectors = self.paragraph_embeddings.get(doc_id)

            if paragraph_vectors is None or len(paragraph_vectors) == 0:
                result[doc_id] = np.empty((0, 0), dtype=np.float32)
                continue

            paragraph_vectors = np.asarray(paragraph_vectors, dtype=np.float32)

            directions: List[np.ndarray] = []

            # d_1 = normalize(p_1 - 0)
            first_direction = self._normalize_dense(paragraph_vectors[0])
            if first_direction is not None:
                directions.append(first_direction)

            # d_i = normalize(p_i - p_{i-1})
            for i in range(1, len(paragraph_vectors)):
                direction = paragraph_vectors[i] - paragraph_vectors[i - 1]
                direction = self._normalize_dense(direction)

                if direction is not None:
                    directions.append(direction)

            if directions:
                result[doc_id] = np.vstack(directions).astype(np.float32)
            else:
                dim = paragraph_vectors.shape[1]
                result[doc_id] = np.empty((0, dim), dtype=np.float32)

        return result

    def _intention_score(
        self,
        query_intent: np.ndarray,
        doc_id: int,
    ) -> float:
        doc_directions = self.doc_intent_vectors.get(doc_id)

        if doc_directions is None or len(doc_directions) == 0:
            return 0.0

        if doc_directions.shape[1] != query_intent.shape[0]:
            raise ValueError(
                "Dimension mismatch between query embedding and paragraph embeddings. "
                f"query dim={query_intent.shape[0]}, "
                f"doc dim={doc_directions.shape[1]}. "
                "Use the same SentenceTransformer model for both."
            )

        similarities = doc_directions @ query_intent

        if similarities.size == 0:
            return 0.0

        k = min(self.intent_k, similarities.size)
        top_values = np.sort(similarities)[-k:][::-1]

        return self._softmax_weighted_average(
            values=top_values,
            temperature=self.temperature,
        )

    @staticmethod
    def _softmax_weighted_average(
        values: np.ndarray,
        temperature: float,
    ) -> float:
        if values.size == 0:
            return 0.0

        scaled = values / temperature
        scaled = scaled - np.max(scaled)

        exp_values = np.exp(scaled)
        exp_sum = np.sum(exp_values)

        if exp_sum == 0.0:
            return 0.0

        weights = exp_values / exp_sum

        return float(np.sum(weights * values))

    @staticmethod
    def _normalize_dense(vector: np.ndarray):
        norm = float(np.linalg.norm(vector))

        if norm == 0.0:
            return None

        return (vector / norm).astype(np.float32)

    @classmethod
    def _normalize_dense_or_zero(cls, vector: np.ndarray) -> np.ndarray:
        normalized = cls._normalize_dense(vector)

        if normalized is None:
            return np.zeros_like(vector, dtype=np.float32)

        return normalized

    @staticmethod
    def _minmax_normalize(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score - min_score == 0.0:
            return {
                doc_id: 0.0
                for doc_id in scores.keys()
            }

        return {
            doc_id: (score - min_score) / (max_score - min_score)
            for doc_id, score in scores.items()
        }