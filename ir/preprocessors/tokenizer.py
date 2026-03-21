import re
from typing import Iterable, List, Optional, Set


class Tokenizer:
    """
    A simple reusable tokenizer for IR/NLP coursework.

    Features:
    - lowercasing
    - punctuation removal
    - optional number removal
    - optional stopword removal
    - whitespace tokenization
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[Iterable[str]] = None,
        min_token_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stopwords: Set[str] = set(stopwords) if stopwords else set()
        self.min_token_length = min_token_length

    def normalize(self, text: str) -> str:
        """
        Normalize raw text before tokenization.
        """
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if self.lowercase:
            text = text.lower()

        # Keep letters, numbers, and whitespace only.
        # Replace punctuation/symbols with spaces.
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Convert a raw string into a list of tokens.
        """
        normalized_text = self.normalize(text)

        if not normalized_text:
            return []

        tokens = normalized_text.split()

        # Remove short tokens
        if self.min_token_length > 1:
            tokens = [token for token in tokens if len(token) >= self.min_token_length]

        # Remove stopwords if enabled
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def batch_tokenize(self, texts: Iterable[str]) -> List[List[str]]:
        """
        Tokenize multiple texts.
        """
        return [self.tokenize(text) for text in texts]