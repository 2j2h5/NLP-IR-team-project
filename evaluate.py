import argparse
import os
import pickle
from typing import Any, Dict, Set

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.weighting.tfidf import TFIDFWeighter
from ir.models.vector_space_model import VectorSpaceModel
from ir.models.boolean_model import BooleanModel
from ir.evaluator.evaluator import Evaluator


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_index(path: str) -> InvertedIndex:
    data = load_pickle(path)
    return InvertedIndex.from_dict(data)


def build_model(
    model_name: str,
    index: InvertedIndex,
    title_weight: float = 2.0,
    body_weight: float = 1.0,
    use_log_tf: bool = True,
    smooth_idf: bool = True,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
):
    tokenizer = Tokenizer(
        lowercase=True,
        remove_numbers=remove_numbers,
        remove_stopwords=remove_stopwords,
        stopwords=None,
        min_token_length=min_token_length,
    )

    if model_name == "vsm":
        weighter = TFIDFWeighter(
            title_weight=title_weight,
            body_weight=body_weight,
            use_log_tf=use_log_tf,
            smooth_idf=smooth_idf,
        )
        model = VectorSpaceModel(
            index=index,
            tokenizer=tokenizer,
            weighter=weighter,
        )

    elif model_name == "boolean":
        model = BooleanModel(
            index=index,
            tokenizer=tokenizer,
        )

    elif model_name == "link-vsm":
        raise NotImplementedError(
            "link-vsm is not implemented yet. "
            "Implement ir/models/link_aware_vsm.py first."
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.build()
    return model


def get_default_prefix(dataset: str, size: int) -> str:
    if dataset == "cisi":
        return "outputs/cisi"
    if dataset == "kilt":
        return f"outputs/kilt_{size}"
    raise ValueError(f"Unsupported dataset: {dataset}")


def print_summary(results: Dict[str, Any], model_name: str, dataset: str, k: int) -> None:
    print("\n" + "=" * 72)
    print("Final Evaluation Summary")
    print("=" * 72)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Mean Precision@{k}: {results[f'mean_precision@{k}']:.4f}")
    print(f"Mean Recall@{k}   : {results[f'mean_recall@{k}']:.4f}")
    print(f"Mean F0.25@{k}    : {results[f'mean_f0.25@{k}']:.4f}")
    print(f"Mean F0.5@{k}     : {results[f'mean_f0.5@{k}']:.4f}")
    print(f"Mean F1@{k}       : {results[f'mean_f1@{k}']:.4f}")
    print(f"Mean F2@{k}       : {results[f'mean_f2@{k}']:.4f}")
    print(f"Mean F4@{k}       : {results[f'mean_f4@{k}']:.4f}")
    print(f"MAP              : {results['map']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval models on CISI or KILT-Wikipedia artifacts."
    )

    parser.add_argument(
        "--dataset",
        choices=["cisi", "kilt"],
        default="cisi",
        help="Dataset artifacts to evaluate.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=500,
        help="KILT sample size used in artifact names. Used only when --dataset kilt.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Artifact prefix. Example: outputs/cisi or outputs/kilt_500.",
    )

    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Optional explicit path to index pickle.",
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=None,
        help="Optional explicit path to queries pickle.",
    )
    parser.add_argument(
        "--relevance",
        type=str,
        default=None,
        help="Optional explicit path to relevance pickle.",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Optional explicit path to graph pickle. Used later for link-aware models.",
    )

    parser.add_argument(
        "--model",
        choices=["vsm", "boolean", "link-vsm"],
        default="vsm",
        help="Retrieval model to evaluate.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff rank k for evaluation.",
    )

    parser.add_argument(
        "--title-weight",
        type=float,
        default=2.0,
        help="Weight for title field TF. Used by VSM.",
    )
    parser.add_argument(
        "--body-weight",
        type=float,
        default=1.0,
        help="Weight for body field TF. Used by VSM.",
    )
    parser.add_argument(
        "--no-log-tf",
        action="store_true",
        help="Disable log-scaled TF. Used by VSM.",
    )
    parser.add_argument(
        "--no-smooth-idf",
        action="store_true",
        help="Disable smoothed IDF. Used by VSM.",
    )

    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove numeric tokens during query tokenization.",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords during query tokenization.",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=1,
        help="Minimum token length to keep during query tokenization.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-query metric output.",
    )

    args = parser.parse_args()

    prefix = args.prefix or get_default_prefix(args.dataset, args.size)

    index_path = args.index or f"{prefix}_index.pkl"
    queries_path = args.queries or f"{prefix}_queries.pkl"
    relevance_path = args.relevance or f"{prefix}_relevance.pkl"
    graph_path = args.graph or f"{prefix}_graph.pkl"

    print(f"Dataset        : {args.dataset}")
    print(f"Artifact prefix: {prefix}")
    print(f"Index path     : {index_path}")
    print(f"Queries path   : {queries_path}")
    print(f"Relevance path : {relevance_path}")

    if args.model == "link-vsm":
        print(f"Graph path     : {graph_path}")

    print("Loading index...")
    index = load_index(index_path)

    print("Loading queries and relevance...")
    queries: Dict[int, str] = load_pickle(queries_path)
    relevance: Dict[int, Set[int]] = load_pickle(relevance_path)

    print(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        index=index,
        title_weight=args.title_weight,
        body_weight=args.body_weight,
        use_log_tf=not args.no_log_tf,
        smooth_idf=not args.no_smooth_idf,
        remove_numbers=args.remove_numbers,
        remove_stopwords=args.remove_stopwords,
        min_token_length=args.min_token_length,
    )

    print(f"Indexed documents : {len(index)}")
    print(f"Vocabulary size   : {len(index.vocabulary())}")
    print(f"Queries           : {len(queries)}")
    print(f"Relevance sets    : {len(relevance)}")
    print(f"Evaluating at k   : {args.top_k}")

    evaluator = Evaluator(model)
    results = evaluator.evaluate_all(
        queries=queries,
        relevance_judgments=relevance,
        k=args.top_k,
        verbose=not args.quiet,
    )

    print_summary(
        results=results,
        model_name=args.model,
        dataset=args.dataset,
        k=args.top_k,
    )


if __name__ == "__main__":
    main()