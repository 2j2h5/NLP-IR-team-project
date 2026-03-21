import math
from typing import Dict

from ir.indexing.inverted_index import InvertedIndex


class TFIDFWeighter:
    """
    TF-IDF weighting module for a field-aware inverted index.

    Supports:
    - raw TF or log-scaled TF
    - smoothed IDF
    - separate title/body weighting
    """

    def __init__(
        self,
        title_weight: float = 2.0,
        body_weight: float = 1.0,
        use_log_tf: bool = True,
        smooth_idf: bool = True,
    ) -> None:
        self.title_weight = title_weight
        self.body_weight = body_weight
        self.use_log_tf = use_log_tf
        self.smooth_idf = smooth_idf

    def tf(self, raw_tf: int) -> float:
        """
        Compute TF weight from raw term frequency.

        Args:
            raw_tf: raw term frequency

        Returns:
            TF weight
        """
        if raw_tf < 0:
            raise ValueError("raw_tf must be non-negative.")

        if raw_tf == 0:
            return 0.0

        if self.use_log_tf:
            return 1.0 + math.log(raw_tf)

        return float(raw_tf)

    def idf(self, df: int, num_docs: int) -> float:
        """
        Compute IDF weight.

        Args:
            df: document frequency of the term
            num_docs: total number of documents

        Returns:
            IDF weight
        """
        if df < 0:
            raise ValueError("df must be non-negative.")
        if num_docs <= 0:
            raise ValueError("num_docs must be positive.")

        if df == 0:
            return 0.0

        if self.smooth_idf:
            return math.log((num_docs + 1) / (df + 1)) + 1.0

        return math.log(num_docs / df)

    def field_tf(
        self,
        title_tf: int,
        body_tf: int,
    ) -> float:
        """
        Compute field-weighted TF using title/body term frequencies.

        Args:
            title_tf: raw term frequency in title
            body_tf: raw term frequency in body

        Returns:
            weighted TF across fields
        """
        return (
            self.title_weight * self.tf(title_tf)
            + self.body_weight * self.tf(body_tf)
        )

    def term_weight(
        self,
        term: str,
        doc_id: int,
        index: InvertedIndex,
    ) -> float:
        """
        Compute TF-IDF weight of a term in a specific document.

        Args:
            term: term
            doc_id: document ID
            index: InvertedIndex instance

        Returns:
            TF-IDF weight
        """
        title_tf = index.get_title_tf(term, doc_id)
        body_tf = index.get_body_tf(term, doc_id)
        df = index.get_df(term)
        idf_value = self.idf(df, index.num_docs)

        return self.field_tf(title_tf, body_tf) * idf_value

    def query_term_weight(
        self,
        query_tf: int,
        term: str,
        index: InvertedIndex,
    ) -> float:
        """
        Compute TF-IDF weight of a term in a query.

        Args:
            query_tf: raw term frequency in the query
            term: term
            index: InvertedIndex instance

        Returns:
            query-side TF-IDF weight
        """
        df = index.get_df(term)
        idf_value = self.idf(df, index.num_docs)

        return self.tf(query_tf) * idf_value

    def document_vector(
        self,
        doc_id: int,
        index: InvertedIndex,
    ) -> Dict[str, float]:
        """
        Build a sparse TF-IDF vector for a document.

        Args:
            doc_id: document ID
            index: InvertedIndex instance

        Returns:
            {term: tf-idf weight}
        """
        vector: Dict[str, float] = {}

        for term in index.vocabulary():
            weight = self.term_weight(term, doc_id, index)
            if weight > 0.0:
                vector[term] = weight

        return vector

    def query_vector(
        self,
        query_term_counts: Dict[str, int],
        index: InvertedIndex,
    ) -> Dict[str, float]:
        """
        Build a sparse TF-IDF vector for a query.

        Args:
            query_term_counts: {term: raw query tf}
            index: InvertedIndex instance

        Returns:
            {term: tf-idf weight}
        """
        vector: Dict[str, float] = {}

        for term, q_tf in query_term_counts.items():
            weight = self.query_term_weight(q_tf, term, index)
            if weight > 0.0:
                vector[term] = weight

        return vector