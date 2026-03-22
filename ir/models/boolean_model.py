from typing import List, Set, Tuple

from ir.indexing.inverted_index import InvertedIndex
from ir.preprocessors.tokenizer import Tokenizer


class BooleanModel:
    def __init__(self, index: InvertedIndex, tokenizer: Tokenizer) -> None:
        self.index = index
        self.tokenizer = tokenizer

    def _all_doc_ids(self) -> Set[int]:
        return set(self.index.documents.keys())

    def _term_docs(self, term: str) -> Set[int]:
        postings = self.index.get_postings(term)
        if not postings:
            return set()

        # field-aware postings 지원
        if isinstance(postings, dict) and ("title" in postings or "body" in postings):
            doc_ids = set()
            doc_ids.update(postings.get("title", {}).keys())
            doc_ids.update(postings.get("body", {}).keys())
            return doc_ids

        return set(postings.keys())

    def search(self, query: str) -> List[Tuple[int, float]]:
        tokens = query.split()
        normalized_tokens = []

        for token in tokens:
            upper = token.upper()
            if upper in {"AND", "OR", "NOT"}:
                normalized_tokens.append(upper)
            else:
                term_tokens = self.tokenizer.tokenize(token)
                normalized_tokens.extend(term_tokens)

        if not normalized_tokens:
            return []

        result = None
        op = "AND"   # 기본 연산자
        negate_next = False

        for token in normalized_tokens:
            if token == "NOT":
                negate_next = True
                continue

            if token in {"AND", "OR"}:
                op = token
                continue

            docs = self._term_docs(token)

            if negate_next:
                docs = self._all_doc_ids() - docs
                negate_next = False

            if result is None:
                result = docs
            elif op == "AND":
                result &= docs
            elif op == "OR":
                result |= docs

        if result is None:
            return []

        return [(doc_id, 1.0) for doc_id in sorted(result)]