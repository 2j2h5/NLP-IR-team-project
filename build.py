import argparse
import os
import pickle
from typing import Dict

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex


def parse_cisi_all(file_path: str) -> Dict[int, Dict[str, str]]:
    """
    Parse the CISI.ALL file into a fielded document dictionary.

    Expected output format:
    {
        doc_id: {
            "title": "...",
            "body": "..."
        }
    }
    """
    documents: Dict[int, Dict[str, str]] = {}

    current_doc_id = None
    current_section = None

    title_lines = []
    body_lines = []

    def save_current_document() -> None:
        nonlocal current_doc_id, title_lines, body_lines

        if current_doc_id is None:
            return

        documents[current_doc_id] = {
            "title": " ".join(title_lines).strip(),
            "body": " ".join(body_lines).strip(),
        }

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith(".I "):
                save_current_document()

                current_doc_id = int(line.split()[1])
                current_section = None
                title_lines = []
                body_lines = []

            elif line.startswith(".T"):
                current_section = "title"

            elif line.startswith(".W"):
                current_section = "body"

            elif line.startswith(".A") or line.startswith(".B") or line.startswith(".X"):
                current_section = None

            else:
                if current_section == "title":
                    title_lines.append(line.strip())
                elif current_section == "body":
                    body_lines.append(line.strip())

    save_current_document()
    return documents


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