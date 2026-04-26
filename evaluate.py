import argparse
import os
import pickle
import sys
from typing import Dict, Set

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.weighting.tfidf import TFIDFWeighter
from ir.models.vector_space_model import VectorSpaceModel
from ir.models.boolean_model import BooleanModel
from ir.evaluator.evaluator import Evaluator
from ir.datasets.cisi import parse_cisi_queries, parse_cisi_rel


def load_index(index_path: str) -> InvertedIndex:
    with open(index_path, "rb") as f:
        data = pickle.load(f)

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

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.build()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval models on CISI queries/relevance."
    )

    parser.add_argument(
        "--index",
        type=str,
        default="outputs/index.pkl",
        help="Path to saved index pickle file",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        required=True,
        help="Path to CISI.QRY file",
    )
    parser.add_argument(
        "--rel-file",
        type=str,
        required=True,
        help="Path to CISI.REL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vsm", "boolean"],
        default="vsm",
        help="Retrieval model to evaluate",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff rank k for Precision@k and Recall@k",
    )

    parser.add_argument(
        "--title-weight",
        type=float,
        default=2.0,
        help="Weight for title field TF. Used only for VSM.",
    )
    parser.add_argument(
        "--body-weight",
        type=float,
        default=1.0,
        help="Weight for body field TF. Used only for VSM.",
    )
    parser.add_argument(
        "--no-log-tf",
        action="store_true",
        help="Disable log-scaled TF. Used only for VSM.",
    )
    parser.add_argument(
        "--no-smooth-idf",
        action="store_true",
        help="Disable smoothed IDF. Used only for VSM.",
    )
    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove numeric tokens during query tokenization",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords during query tokenization",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=1,
        help="Minimum token length to keep during query tokenization",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-query metric output",
    )

    args = parser.parse_args()

    print(f"Loading index from: {args.index}")
    index = load_index(args.index)

    print(f"Building {args.model} model...")
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

    print(f"Parsing queries from: {args.query_file}")
    queries = parse_cisi_queries(args.query_file)

    print(f"Parsing relevance from: {args.rel_file}")
    relevance = parse_cisi_rel(args.rel_file)

    print(f"Model             : {args.model}")
    print(f"Number of queries : {len(queries)}")
    print(f"Queries with labels: {len(relevance)}")
    print(f"Evaluating at k   : {args.top_k}")

    evaluator = Evaluator(model)
    results = evaluator.evaluate_all(
        queries=queries,
        relevance_judgments=relevance,
        k=args.top_k,
        verbose=not args.quiet,
    )

    print("\n" + "=" * 72)
    print("Final Evaluation Summary")
    print("=" * 72)
    print(f"Model: {args.model}")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Mean Precision@{args.top_k}: {results[f'mean_precision@{args.top_k}']:.4f}")
    print(f"Mean Recall@{args.top_k}   : {results[f'mean_recall@{args.top_k}']:.4f}")
    print(f"Mean F0.25@{args.top_k}    : {results[f'mean_f0.25@{args.top_k}']:.4f}")
    print(f"Mean F0.5@{args.top_k}     : {results[f'mean_f0.5@{args.top_k}']:.4f}")
    print(f"Mean F1@{args.top_k}       : {results[f'mean_f1@{args.top_k}']:.4f}")
    print(f"Mean F2@{args.top_k}       : {results[f'mean_f2@{args.top_k}']:.4f}")
    print(f"Mean F4@{args.top_k}       : {results[f'mean_f4@{args.top_k}']:.4f}")
    print(f"MAP                        : {results['map']:.4f}")


if __name__ == "__main__":
    main()