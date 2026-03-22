import re
from pathlib import Path
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
    - default stopwords loading from data/stopwords.txt
    """

    DEFAULT_STOPWORDS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "stopwords.txt"

    def __init__(
        self,
        lowercase: bool = True,
        remove_numbers: bool = False,
        remove_stopwords: bool = False,
        stopwords: Optional[Iterable[str]] = None,
        stopwords_path: Optional[str] = None,
        min_token_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length

        self.stopwords: Set[str] = self._initialize_stopwords(
            stopwords=stopwords,
            stopwords_path=stopwords_path,
        )

    @classmethod
    def _load_stopwords_from_file(cls, path: Path) -> Set[str]:
        """
        Load stopwords from a text file, one word per line.
        Empty lines are ignored.
        """
        if not path.exists():
            return set()

        with path.open("r", encoding="utf-8") as f:
            return {line.strip().lower() for line in f if line.strip()}

    @classmethod
    def _initialize_stopwords(
        cls,
        stopwords: Optional[Iterable[str]],
        stopwords_path: Optional[str],
    ) -> Set[str]:
        """
        Stopword priority:
        1. Explicit stopwords iterable
        2. Explicit stopwords_path
        3. Default data/stopwords.txt
        """
        if stopwords is not None:
            return {word.lower() for word in stopwords}

        if stopwords_path is not None:
            return cls._load_stopwords_from_file(Path(stopwords_path))

        return cls._load_stopwords_from_file(cls.DEFAULT_STOPWORDS_PATH)

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