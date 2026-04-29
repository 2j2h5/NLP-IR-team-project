import argparse
import os
import pickle
from typing import Any, Dict

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.datasets.cisi import parse_cisi_all, parse_cisi_queries, parse_cisi_rel
from ir.datasets.kilt_wikipedia import build_kilt_subset


def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_index(
    documents: Dict[int, Dict[str, str]],
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
) -> InvertedIndex:
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


def build_cisi(args: argparse.Namespace) -> None:
    print(f"Parsing CISI documents from: {args.input}")
    documents = parse_cisi_all(args.input)

    print("Building CISI index...")
    index = build_index(
        documents=documents,
        remove_numbers=args.remove_numbers,
        remove_stopwords=args.remove_stopwords,
        min_token_length=args.min_token_length,
    )

    print(f"Parsing CISI queries from: {args.query_file}")
    queries = parse_cisi_queries(args.query_file)

    print(f"Parsing CISI relevance from: {args.rel_file}")
    relevance = parse_cisi_rel(args.rel_file)

    prefix = args.output_prefix

    save_pickle(index.to_dict(), f"{prefix}_index.pkl")
    save_pickle(queries, f"{prefix}_queries.pkl")
    save_pickle(relevance, f"{prefix}_relevance.pkl")

    print("CISI build completed.")
    print(f"Documents       : {len(index)}")
    print(f"Vocabulary size : {len(index.vocabulary())}")
    print(f"Queries         : {len(queries)}")
    print(f"Relevance sets  : {len(relevance)}")
    print(f"Saved prefix    : {prefix}")


def build_kilt(args: argparse.Namespace) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for KILT-Wikipedia. "
            "Install it with: pip install datasets"
        ) from exc

    print(f"Loading Hugging Face dataset: {args.hf_dataset}")
    dataset = load_dataset(
        args.hf_dataset,
        split=args.hf_split,
        streaming=args.streaming,
        trust_remote_code=True,
    )

    print("Building KILT-Wikipedia subset...")
    print(f"Target size    : {args.target_size}")
    print(f"Max depth      : {args.max_depth}")
    print(f"Load limit     : {args.load_limit}")
    print(f"Num seeds      : {args.num_auto_seeds}")
    print(f"Seed strategy  : {args.seed_strategy}")
    print(f"Random seed    : {args.random_seed}")
    print(f"Max queries    : {args.max_queries}")

    subset = build_kilt_subset(
        records=dataset,
        target_size=args.target_size,
        max_depth=args.max_depth,
        load_limit=args.load_limit,
        random_seed=args.random_seed,
        num_auto_seeds=args.num_auto_seeds,
        seed_strategy=args.seed_strategy,
        max_queries=args.max_queries,
    )

    documents = subset["documents"]
    edges = subset["edges"]
    queries = subset["queries"]
    relevance = subset["relevance"]
    meta = subset["meta"]

    print("Building KILT index...")
    index = build_index(
        documents=documents,
        remove_numbers=args.remove_numbers,
        remove_stopwords=args.remove_stopwords,
        min_token_length=args.min_token_length,
    )

    prefix = args.output_prefix or f"outputs/kilt_{args.target_size}"

    save_pickle(index.to_dict(), f"{prefix}_index.pkl")
    save_pickle(edges, f"{prefix}_graph.pkl")
    save_pickle(queries, f"{prefix}_queries.pkl")
    save_pickle(relevance, f"{prefix}_relevance.pkl")
    save_pickle(meta, f"{prefix}_meta.pkl")

    print("KILT build completed.")
    print(f"Documents       : {len(index)}")
    print(f"Vocabulary size : {len(index.vocabulary())}")
    print(f"Edges           : {len(edges)}")
    print(f"Queries         : {len(queries)}")
    print(f"Relevance sets  : {len(relevance)}")
    print(f"Saved prefix    : {prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IR dataset artifacts for CISI or KILT-Wikipedia."
    )

    parser.add_argument(
        "--dataset",
        choices=["cisi", "kilt"],
        default="cisi",
        help="Dataset to build.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Output prefix without suffix. "
            "Example: outputs/cisi or outputs/kilt_1000_highdeg_q150"
        ),
    )

    # CISI options
    parser.add_argument(
        "--input",
        type=str,
        default="data/CISI.ALL",
        help="Path to CISI.ALL. Used only when --dataset cisi.",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        default="data/CISI.QRY",
        help="Path to CISI.QRY. Used only when --dataset cisi.",
    )
    parser.add_argument(
        "--rel-file",
        type=str,
        default="data/CISI.REL",
        help="Path to CISI.REL. Used only when --dataset cisi.",
    )

    # KILT options
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="facebook/kilt_wikipedia",
        help="Hugging Face dataset name. Used only when --dataset kilt.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="full",
        help="Hugging Face split name. Used only when --dataset kilt.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=500,
        help="Target number of sampled KILT documents.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum BFS depth for KILT sampling.",
    )
    parser.add_argument(
        "--load-limit",
        type=int,
        default=100000,
        help="Maximum number of KILT records to load before sampling.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use Hugging Face streaming mode for KILT.",
    )
    parser.add_argument(
        "--num-auto-seeds",
        type=int,
        default=50,
        help="Number of BFS seed documents for KILT.",
    )
    parser.add_argument(
        "--seed-strategy",
        type=str,
        default="high_outdegree",
        choices=["high_outdegree", "random"],
        help=(
            "Seed selection strategy for KILT BFS sampling. "
            "'high_outdegree' selects documents with large internal out-degree. "
            "'random' selects seed documents randomly."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and BFS neighbor ordering.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of generated KILT pseudo-queries.",
    )

    # Tokenization options
    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove numeric tokens during tokenization.",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords during tokenization.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=1,
        help="Minimum token length to keep.",
    )

    args = parser.parse_args()

    if args.output_prefix is None:
        if args.dataset == "cisi":
            args.output_prefix = "outputs/cisi"
        else:
            args.output_prefix = f"outputs/kilt_{args.target_size}"

    if args.dataset == "cisi":
        build_cisi(args)
    elif args.dataset == "kilt":
        build_kilt(args)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__ == "__main__":
    main()