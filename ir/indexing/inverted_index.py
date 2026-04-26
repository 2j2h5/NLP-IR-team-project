from collections import Counter, defaultdict
from typing import Dict, List, Set


class InvertedIndex:
    """
    Field-aware inverted index for IR coursework.

    Stores term frequencies separately for title and body fields.
    Also stores document-level term sets for efficient sparse vector construction.
    """

    def __init__(self) -> None:
        self.postings: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(dict)
        self.doc_lengths: Dict[int, Dict[str, int]] = {}
        self.documents: Dict[int, Dict[str, str]] = {}

        # doc_id -> set of terms appearing in either title or body
        self.doc_terms: Dict[int, Set[str]] = {}

        self.num_docs: int = 0

    def add_document(
        self,
        doc_id: int,
        title: str,
        body: str,
        tokenizer
    ) -> None:
        if doc_id in self.documents:
            raise ValueError(f"Document ID {doc_id} already exists in the index.")

        title_tokens = tokenizer.tokenize(title)
        body_tokens = tokenizer.tokenize(body)

        self.documents[doc_id] = {
            "title": title,
            "body": body,
        }

        self.doc_lengths[doc_id] = {
            "title": len(title_tokens),
            "body": len(body_tokens),
        }

        self.num_docs += 1

        title_counts = Counter(title_tokens)
        body_counts = Counter(body_tokens)

        all_terms = set(title_counts.keys()) | set(body_counts.keys())
        self.doc_terms[doc_id] = all_terms

        for term in all_terms:
            self.postings[term][doc_id] = {
                "title": title_counts.get(term, 0),
                "body": body_counts.get(term, 0),
            }

    def add_documents(
        self,
        documents: Dict[int, Dict[str, str]],
        tokenizer
    ) -> None:
        for doc_id, fields in documents.items():
            if "title" not in fields or "body" not in fields:
                raise ValueError(
                    f"Document {doc_id} must contain both 'title' and 'body'."
                )

            self.add_document(
                doc_id=doc_id,
                title=fields["title"],
                body=fields["body"],
                tokenizer=tokenizer,
            )

    def get_postings(self, term: str) -> Dict[int, Dict[str, int]]:
        return self.postings.get(term, {})

    def get_df(self, term: str) -> int:
        return len(self.get_postings(term))

    def get_tf(self, term: str, doc_id: int, field: str = "body") -> int:
        if field not in {"title", "body"}:
            raise ValueError("field must be either 'title' or 'body'.")

        return self.postings.get(term, {}).get(doc_id, {}).get(field, 0)

    def get_title_tf(self, term: str, doc_id: int) -> int:
        return self.get_tf(term, doc_id, field="title")

    def get_body_tf(self, term: str, doc_id: int) -> int:
        return self.get_tf(term, doc_id, field="body")

    def get_doc_terms(self, doc_id: int) -> Set[str]:
        """
        Return the set of terms that appear in the given document.

        This is used to build sparse document vectors efficiently without
        scanning the entire vocabulary.
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document ID {doc_id} not found.")

        return self.doc_terms.get(doc_id, set())

    def get_doc_length(self, doc_id: int, field: str = "body") -> int:
        if field not in {"title", "body"}:
            raise ValueError("field must be either 'title' or 'body'.")

        if doc_id not in self.doc_lengths:
            raise KeyError(f"Document ID {doc_id} not found.")

        return self.doc_lengths[doc_id][field]

    def get_document(self, doc_id: int) -> Dict[str, str]:
        if doc_id not in self.documents:
            raise KeyError(f"Document ID {doc_id} not found.")

        return self.documents[doc_id]

    def get_title(self, doc_id: int) -> str:
        return self.get_document(doc_id)["title"]

    def get_body(self, doc_id: int) -> str:
        return self.get_document(doc_id)["body"]

    def contains_term(self, term: str) -> bool:
        return term in self.postings

    def vocabulary(self) -> List[str]:
        return sorted(self.postings.keys())

    def __len__(self) -> int:
        return self.num_docs

    def to_dict(self) -> Dict:
        return {
            "postings": dict(self.postings),
            "doc_lengths": self.doc_lengths,
            "documents": self.documents,
            "doc_terms": {
                doc_id: list(terms)
                for doc_id, terms in self.doc_terms.items()
            },
            "num_docs": self.num_docs,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InvertedIndex":
        index = cls()
        index.postings = defaultdict(dict, data["postings"])
        index.doc_lengths = data["doc_lengths"]
        index.documents = data["documents"]
        index.num_docs = data["num_docs"]

        # Backward compatibility for old index.pkl files
        if "doc_terms" in data:
            index.doc_terms = {
                int(doc_id): set(terms)
                for doc_id, terms in data["doc_terms"].items()
            }
        else:
            index.doc_terms = defaultdict(set)
            for term, postings in index.postings.items():
                for doc_id in postings.keys():
                    index.doc_terms[int(doc_id)].add(term)
            index.doc_terms = dict(index.doc_terms)

        return index