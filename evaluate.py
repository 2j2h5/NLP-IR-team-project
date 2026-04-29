import argparse
import csv
import os
import pickle
from typing import Any, Dict, Set

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.weighting.tfidf import TFIDFWeighter
from ir.models.vector_space_model import VectorSpaceModel
from ir.models.boolean_model import BooleanModel
from ir.models.link_aware_vsm import LinkAwareVectorSpaceModel
from ir.graph.link_graph import LinkGraph
from ir.evaluator.evaluator import Evaluator


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_index(path: str) -> InvertedIndex:
    data = load_pickle(path)
    return InvertedIndex.from_dict(data)


def get_default_prefix(dataset: str, size: int) -> str:
    if dataset == "cisi":
        return "outputs/cisi"
    if dataset == "kilt":
        return f"outputs/kilt_{size}"
    raise ValueError(f"Unsupported dataset: {dataset}")


def append_csv(csv_path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "prefix",
        "top_k",
        "link_score",
        "alpha",
        "seed_strategy",
        "random_seed",
        "num_docs",
        "vocab_size",
        "num_queries",
        "precision",
        "recall",
        "f0.25",
        "f0.5",
        "f1",
        "f2",
        "f4",
        "map",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


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
    graph: LinkGraph = None,
    link_score: str = "pagerank",
    alpha: float = 0.8,
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
        if graph is None:
            raise ValueError("link-vsm requires a link graph.")

        weighter = TFIDFWeighter(
            title_weight=title_weight,
            body_weight=body_weight,
            use_log_tf=use_log_tf,
            smooth_idf=smooth_idf,
        )

        model = LinkAwareVectorSpaceModel(
            index=index,
            tokenizer=tokenizer,
            weighter=weighter,
            graph=graph,
            link_score=link_score,
            alpha=alpha,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.build()
    return model


def print_summary(
    results: Dict[str, Any],
    model_name: str,
    dataset: str,
    k: int,
    link_score: str = None,
    alpha: float = None,
) -> None:
    print("\n" + "=" * 72)
    print("Final Evaluation Summary")
    print("=" * 72)
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")

    if model_name == "link-vsm":
        print(f"Link score: {link_score}")
        print(f"Alpha: {alpha}")

    print(f"Number of queries: {results['num_queries']}")
    print(f"Mean Precision@{k}: {results[f'mean_precision@{k}']:.4f}")
    print(f"Mean Recall@{k}   : {results[f'mean_recall@{k}']:.4f}")
    print(f"Mean F0.25@{k}    : {results[f'mean_f0.25@{k}']:.4f}")
    print(f"Mean F0.5@{k}     : {results[f'mean_f0.5@{k}']:.4f}")
    print(f"Mean F1@{k}       : {results[f'mean_f1@{k}']:.4f}")
    print(f"Mean F2@{k}       : {results[f'mean_f2@{k}']:.4f}")
    print(f"Mean F4@{k}       : {results[f'mean_f4@{k}']:.4f}")
    print(f"MAP               : {results['map']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval models on CISI or KILT-Wikipedia artifacts."
    )

    parser.add_argument("--dataset", choices=["cisi", "kilt"], default="cisi")
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--prefix", type=str, default=None)

    parser.add_argument("--index", type=str, default=None)
    parser.add_argument("--queries", type=str, default=None)
    parser.add_argument("--relevance", type=str, default=None)
    parser.add_argument("--graph", type=str, default=None)

    parser.add_argument(
        "--model",
        choices=["vsm", "boolean", "link-vsm"],
        default="vsm",
    )
    parser.add_argument("--top-k", type=int, default=10)

    parser.add_argument(
        "--link-score",
        choices=["indegree", "pagerank"],
        default="pagerank",
    )
    parser.add_argument("--alpha", type=float, default=0.8)

    parser.add_argument("--title-weight", type=float, default=2.0)
    parser.add_argument("--body-weight", type=float, default=1.0)
    parser.add_argument("--no-log-tf", action="store_true")
    parser.add_argument("--no-smooth-idf", action="store_true")

    parser.add_argument("--remove-numbers", action="store_true")
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--min-token-length", type=int, default=1)

    parser.add_argument("--quiet", action="store_true")

    # CSV 저장 옵션
    parser.add_argument("--save-csv", action="store_true")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="outputs/summary/all_results.csv",
    )

    # 실험 메타데이터
    parser.add_argument(
        "--seed-strategy",
        type=str,
        default="",
        choices=["", "high_outdegree", "random"],
    )
    parser.add_argument("--random-seed", type=str, default="")

    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

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
        print(f"Link score     : {args.link_score}")
        print(f"Alpha          : {args.alpha}")

    print("Loading index...")
    index = load_index(index_path)

    print("Loading queries and relevance...")
    queries: Dict[int, str] = load_pickle(queries_path)
    relevance: Dict[int, Set[int]] = load_pickle(relevance_path)

    graph = None
    if args.model == "link-vsm":
        print("Loading graph...")
        edges = load_pickle(graph_path)
        graph = LinkGraph.from_edges(edges)
        print(f"Graph nodes     : {graph.num_nodes()}")
        print(f"Graph edges     : {graph.num_edges()}")

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
        graph=graph,
        link_score=args.link_score,
        alpha=args.alpha,
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
        link_score=args.link_score,
        alpha=args.alpha,
    )

    if args.save_csv:
        k = args.top_k

        row = {
            "dataset": args.dataset,
            "model": args.model,
            "prefix": prefix,
            "top_k": k,
            "link_score": args.link_score if args.model == "link-vsm" else "",
            "alpha": args.alpha if args.model == "link-vsm" else "",
            "seed_strategy": args.seed_strategy,
            "random_seed": args.random_seed,
            "num_docs": len(index),
            "vocab_size": len(index.vocabulary()),
            "num_queries": results["num_queries"],
            "precision": results[f"mean_precision@{k}"],
            "recall": results[f"mean_recall@{k}"],
            "f0.25": results[f"mean_f0.25@{k}"],
            "f0.5": results[f"mean_f0.5@{k}"],
            "f1": results[f"mean_f1@{k}"],
            "f2": results[f"mean_f2@{k}"],
            "f4": results[f"mean_f4@{k}"],
            "map": results["map"],
        }

        append_csv(args.csv_path, row)
        print(f"\n[INFO] Saved CSV row to: {args.csv_path}")


if __name__ == "__main__":
    main()