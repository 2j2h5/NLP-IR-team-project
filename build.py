import argparse
import os
import pickle

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.datasets.cisi import parse_cisi_all

def build_index(
    cisi_all_path: str,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
) -> InvertedIndex:
    """
    Build an inverted index from CISI.ALL.
    """
    documents = parse_cisi_all(cisi_all_path)

    tokenizer = Tokenizer(
        lowercase=True,
        remove_numbers=remove_numbers,
        remove_stopwords=remove_stopwords,
        stopwords=None,
        min_token_length=min_token_length,
    )

    index = InvertedIndex()
    index.add_documents(documents, tokenizer)

    return index


def save_index(index: InvertedIndex, output_path: str) -> None:
    """
    Save the inverted index to a pickle file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(index.to_dict(), f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build inverted index from CISI.ALL")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to CISI.ALL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/index.pkl",
        help="Path to save the built index pickle",
    )
    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove numeric tokens during tokenization",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords during tokenization",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=1,
        help="Minimum token length to keep",
    )

    args = parser.parse_args()

    print(f"Parsing and indexing documents from: {args.input}")
    index = build_index(
        cisi_all_path=args.input,
        remove_numbers=args.remove_numbers,
        remove_stopwords=args.remove_stopwords,
        min_token_length=args.min_token_length,
    )

    save_index(index, args.output)

    print("Build completed.")
    print(f"Number of indexed documents: {len(index)}")
    print(f"Vocabulary size: {len(index.vocabulary())}")
    print(f"Saved index to: {args.output}")


if __name__ == "__main__":
    main()