import argparse
import os
import pickle
import random
import sys
from typing import Dict, List, Tuple, Union

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ir.preprocessors.tokenizer import Tokenizer
from ir.indexing.inverted_index import InvertedIndex
from ir.weighting.tfidf import TFIDFWeighter
from ir.models.vector_space_model import VectorSpaceModel
from ir.models.boolean_model import BooleanModel


def load_index(index_path: str) -> InvertedIndex:
    """
    Load a saved inverted index from a pickle file.py
    """
    with open(index_path, "rb") as f:
        data = pickle.load(f)

    return InvertedIndex.from_dict(data)


def parse_cisi_queries(file_path: str) -> Dict[int, str]:
    """
    Parse CISI.QRY file into:
        {query_id: query_text}
    """
    queries: Dict[int, str] = {}

    current_query_id = None
    current_section = None
    query_lines: List[str] = []

    def save_current_query() -> None:
        nonlocal current_query_id, query_lines
        if current_query_id is None:
            return
        queries[current_query_id] = " ".join(query_lines).strip()

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith(".I "):
                save_current_query()
                current_query_id = int(line.split()[1])
                current_section = None
                query_lines = []

            elif line.startswith(".W"):
                current_section = "body"

            elif line.startswith(".A") or line.startswith(".B") or line.startswith(".T") or line.startswith(".X"):
                current_section = None

            else:
                if current_section == "body":
                    query_lines.append(line.strip())

    save_current_query()
    return queries


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
    """
    Build a retrieval model from a loaded index.
    Supports:
    - vsm
    - boolean
    """
    tokenizer = Tokenizer(
        lowercase=True,
        remove_numbers=remove_numbers,
        remove_stopwords=remove_stopwords,
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
        model.build()
        return model

    if model_name == "boolean":
        return BooleanModel(
            index=index,
            tokenizer=tokenizer,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def resolve_query(args) -> Tuple[str, str]:
    """
    Resolve query input mode.

    Priority:
    1. --query
    2. --query-id + --query-file
    3. --random-query + --query-file
    4. interactive input()

    Returns:
        (query_text, source_description)
    """
    if args.query:
        return args.query, "direct --query input"

    if args.query_id is not None:
        if not args.query_file:
            raise ValueError("--query-id requires --query-file.")
        queries = parse_cisi_queries(args.query_file)
        if args.query_id not in queries:
            raise ValueError(f"Query ID {args.query_id} not found in {args.query_file}.")
        return queries[args.query_id], f"query file (ID={args.query_id})"

    if args.random_query:
        if not args.query_file:
            raise ValueError("--random-query requires --query-file.")
        queries = parse_cisi_queries(args.query_file)
        if not queries:
            raise ValueError(f"No queries found in {args.query_file}.")
        query_id = random.choice(list(queries.keys()))
        return queries[query_id], f"random query from file (ID={query_id})"

    query = input("Enter query: ").strip()
    return query, "interactive input"


def format_snippet(text: str, max_length: int = 200) -> str:
    """
    Create a short one-line snippet from body text.
    """
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def print_results(
    results: List[Tuple[int, float]],
    index: InvertedIndex,
    query: str,
    model,
    query_source: str,
    model_name: str,
    show_explain: bool = False,
    show_body: bool = False,
) -> None:
    """
    Print ranked retrieval results.
    """
    print("=" * 72)
    print(f"Query       : {query}")
    print(f"Query Source: {query_source}")
    print(f"Model       : {model_name}")
    print("=" * 72)

    if not results:
        print("No results found.")
        return

    for rank, (doc_id, score) in enumerate(results, start=1):
        title = index.get_title(doc_id)
        body = index.get_body(doc_id)

        print(f"\nRank   : {rank}")
        print(f"Doc ID : {doc_id}")

        if model_name == "vsm":
            print(f"Score  : {score:.6f}")

        print(f"Title  : {title}")

        if show_body:
            print(f"Snippet: {format_snippet(body)}")

        if show_explain and model_name == "vsm":
            contributions = model.explain(query, doc_id)
            if contributions:
                print("Term Contributions:")
                for term, value in sorted(
                    contributions.items(),
                    key=lambda x: x[1],
                    reverse=True,
                ):
                    print(f"  {term:<20} {value:.6f}")
            else:
                print("Term Contributions: None")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a query using either Boolean retrieval or TF-IDF + cosine similarity VSM."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="outputs/index.pkl",
        help="Path to saved index pickle file",
    )

    # Query input modes
    parser.add_argument(
        "--query",
        type=str,
        help="Direct query string",
    )
    parser.add_argument(
        "--query-file",
        type=str,
        help="Path to CISI.QRY file",
    )
    parser.add_argument(
        "--query-id",
        type=int,
        help="Specific query ID to load from query file",
    )
    parser.add_argument(
        "--random-query",
        action="store_true",
        help="Select a random query from query file",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to return",
    )
    parser.add_argument(
        "--title-weight",
        type=float,
        default=2.0,
        help="Weight for title field TF",
    )
    parser.add_argument(
        "--body-weight",
        type=float,
        default=1.0,
        help="Weight for body field TF",
    )
    parser.add_argument(
        "--no-log-tf",
        action="store_true",
        help="Disable log-scaled TF",
    )
    parser.add_argument(
        "--no-smooth-idf",
        action="store_true",
        help="Disable smoothed IDF",
    )
    parser.add_argument(
        "--remove-numbers",
        action="store_true",
        help="Remove numeric tokens in query tokenization",
    )
    parser.add_argument(
        "--remove-stopwords",
        action="store_true",
        help="Remove stopwords in query tokenization",
    )
    parser.add_argument(
        "--min-token-length",
        type=int,
        default=1,
        help="Minimum token length to keep in query tokenization",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Show per-term contribution for each result",
    )
    parser.add_argument(
        "--show-body",
        action="store_true",
        help="Show body snippet for each result",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["vsm", "boolean"],
        default="vsm",
        help="Retrieval model to use",
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

    query_text, query_source = resolve_query(args)

    if not query_text:
        print("Empty query. Exiting.")
        return

    model.build()
    results = model.search(query=query_text, top_k=args.top_k)

    print_results(
        results=results,
        index=index,
        query=query_text,
        model=model,
        query_source=query_source,
        model_name=args.model,
        show_explain=args.explain,
        show_body=args.show_body,
    )


if __name__ == "__main__":
    main()