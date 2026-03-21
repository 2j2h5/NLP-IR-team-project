from collections import Counter, defaultdict
from typing import Dict, List


class InvertedIndex:
    """
    Field-aware inverted index for IR coursework.

    Stores term frequencies separately for title and body fields.

    Attributes:
        postings:
            term -> {
                doc_id -> {
                    "title": title term frequency,
                    "body": body term frequency
                }
            }

        doc_lengths:
            doc_id -> {
                "title": number of title tokens,
                "body": number of body tokens
            }

        documents:
            doc_id -> {
                "title": original title text,
                "body": original body text
            }

        num_docs:
            total number of indexed documents
    """

    def __init__(self) -> None:
        self.postings: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(dict)
        self.doc_lengths: Dict[int, Dict[str, int]] = {}
        self.documents: Dict[int, Dict[str, str]] = {}
        self.num_docs: int = 0

    def add_document(
        self,
        doc_id: int,
        title: str,
        body: str,
        tokenizer
    ) -> None:
        """
        Add a single document to the index.

        Args:
            doc_id: Unique document ID
            title: Document title
            body: Document body
            tokenizer: Tokenizer object with tokenize(text) method
        """
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
        """
        Add multiple documents to the index.

        Args:
            documents:
                {
                    doc_id: {
                        "title": "...",
                        "body": "..."
                    }
                }
            tokenizer: Tokenizer object
        """
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
        """
        Return postings for a term.

        Returns:
            {
                doc_id: {
                    "title": tf in title,
                    "body": tf in body
                }
            }
        """
        return self.postings.get(term, {})

    def get_df(self, term: str) -> int:
        """
        Return document frequency of a term.
        """
        return len(self.get_postings(term))

    def get_tf(self, term: str, doc_id: int, field: str = "body") -> int:
        """
        Return term frequency for a specific term, document, and field.

        Args:
            term: Query term
            doc_id: Document ID
            field: "title" or "body"

        Returns:
            Term frequency, or 0 if absent
        """
        if field not in {"title", "body"}:
            raise ValueError("field must be either 'title' or 'body'.")

        return self.postings.get(term, {}).get(doc_id, {}).get(field, 0)

    def get_title_tf(self, term: str, doc_id: int) -> int:
        """
        Return title term frequency.
        """
        return self.get_tf(term, doc_id, field="title")

    def get_body_tf(self, term: str, doc_id: int) -> int:
        """
        Return body term frequency.
        """
        return self.get_tf(term, doc_id, field="body")

    def get_doc_length(self, doc_id: int, field: str = "body") -> int:
        """
        Return token length for a specific field in a document.
        """
        if field not in {"title", "body"}:
            raise ValueError("field must be either 'title' or 'body'.")

        if doc_id not in self.doc_lengths:
            raise KeyError(f"Document ID {doc_id} not found.")

        return self.doc_lengths[doc_id][field]

    def get_document(self, doc_id: int) -> Dict[str, str]:
        """
        Return the original document fields.
        """
        if doc_id not in self.documents:
            raise KeyError(f"Document ID {doc_id} not found.")

        return self.documents[doc_id]

    def get_title(self, doc_id: int) -> str:
        """
        Return the original title text.
        """
        return self.get_document(doc_id)["title"]

    def get_body(self, doc_id: int) -> str:
        """
        Return the original body text.
        """
        return self.get_document(doc_id)["body"]

    def contains_term(self, term: str) -> bool:
        """
        Check whether a term exists in the index.
        """
        return term in self.postings

    def vocabulary(self) -> List[str]:
        """
        Return the sorted vocabulary list.
        """
        return sorted(self.postings.keys())

    def __len__(self) -> int:
        """
        Return the number of indexed documents.
        """
        return self.num_docs

    def to_dict(self) -> Dict:
        """
        Convert the index into a serializable dictionary.
        """
        return {
            "postings": dict(self.postings),
            "doc_lengths": self.doc_lengths,
            "documents": self.documents,
            "num_docs": self.num_docs,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "InvertedIndex":
        """
        Restore an InvertedIndex instance from a saved dictionary.
        """
        index = cls()
        index.postings = defaultdict(dict, data["postings"])
        index.doc_lengths = data["doc_lengths"]
        index.documents = data["documents"]
        index.num_docs = data["num_docs"]
        return index