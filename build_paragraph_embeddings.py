import argparse
import os
import pickle
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def filter_paragraphs(paragraphs, min_words=20):
    filtered = []

    stop_sections = {
        "Section::::See also.",
        "Section::::Sources.",
        "Section::::External links.",
        "Section::::References.",
        "Section::::Bibliography.",
    }

    for p in paragraphs:
        p = p.strip()

        if not p:
            continue

        if p in stop_sections:
            break

        if p.startswith("Section::::"):
            continue

        if p.startswith("BULLET::::"):
            continue

        if len(p.split()) < min_words:
            continue

        filtered.append(p)

    return filtered


def build_embeddings(
    paragraphs_by_doc: Dict[int, List[str]],
    model_name: str,
    batch_size: int,
    min_words: int,
    normalize_embeddings: bool,
) -> Dict[int, np.ndarray]:
    model = SentenceTransformer(model_name)

    result: Dict[int, np.ndarray] = {}

    doc_ids = sorted(paragraphs_by_doc.keys())

    total_docs = len(doc_ids)
    total_paragraphs = 0

    for idx, doc_id in enumerate(doc_ids, start=1):
        paragraphs = filter_paragraphs(
            paragraphs=paragraphs_by_doc[doc_id],
            min_words=min_words,
        )

        if not paragraphs:
            result[doc_id] = np.empty((0, model.get_sentence_embedding_dimension()))
            continue

        embeddings = model.encode(
            paragraphs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )

        result[doc_id] = embeddings.astype(np.float32)
        total_paragraphs += len(paragraphs)

        if idx % 100 == 0 or idx == total_docs:
            print(
                f"[{idx}/{total_docs}] embedded docs, "
                f"total paragraphs: {total_paragraphs}"
            )

    print(f"Total docs: {total_docs}")
    print(f"Total embedded paragraphs: {total_paragraphs}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dense paragraph embeddings for KILT paragraphs."
    )

    parser.add_argument(
        "--paragraphs",
        type=str,
        required=True,
        help="Path to paragraphs pickle. Example: outputs/kilt_test_paragraphs.pkl",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Output pickle path. "
            "Example: outputs/kilt_test_paragraph_embeddings.pkl"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Remove title-only or very short paragraphs.",
    )
    parser.add_argument(
        "--no-normalize-embeddings",
        action="store_true",
        help="Disable L2 normalization for paragraph embeddings.",
    )

    args = parser.parse_args()

    print(f"Loading paragraphs: {args.paragraphs}")
    paragraphs_by_doc = load_pickle(args.paragraphs)

    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min words: {args.min_words}")
    print(f"Normalize embeddings: {not args.no_normalize_embeddings}")

    embeddings_by_doc = build_embeddings(
        paragraphs_by_doc=paragraphs_by_doc,
        model_name=args.model,
        batch_size=args.batch_size,
        min_words=args.min_words,
        normalize_embeddings=not args.no_normalize_embeddings,
    )

    save_pickle(embeddings_by_doc, args.output)
    print(f"Saved paragraph embeddings: {args.output}")


if __name__ == "__main__":
    main()