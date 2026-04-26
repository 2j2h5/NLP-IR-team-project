from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import random


@dataclass
class KiltDocument:
    doc_id: int
    title: str
    body: str
    first_paragraph: str
    out_links: Set[int]


def normalize_title(title: str) -> str:
    return title.strip().replace("_", " ").lower()


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_paragraphs(record: Dict[str, Any]) -> List[str]:
    text = record.get("text", "")

    if isinstance(text, dict):
        paragraphs = text.get("paragraph", [])
    elif isinstance(text, list):
        paragraphs = text
    elif isinstance(text, str):
        paragraphs = [text]
    else:
        paragraphs = []

    return [p.strip() for p in paragraphs if isinstance(p, str) and p.strip()]


def extract_anchor_target_ids(record: Dict[str, Any]) -> Set[int]:
    anchors = record.get("anchors", {})
    target_ids: Set[int] = set()

    if isinstance(anchors, dict):
        candidates = (
            anchors.get("wikipedia_id")
            or anchors.get("wikipedia_ids")
            or anchors.get("target_wikipedia_id")
            or anchors.get("target_wikipedia_ids")
            or []
        )

        if isinstance(candidates, list):
            for value in candidates:
                target_id = _safe_int(value)
                if target_id is not None:
                    target_ids.add(target_id)

        else:
            target_id = _safe_int(candidates)
            if target_id is not None:
                target_ids.add(target_id)

    elif isinstance(anchors, list):
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue

            value = (
                anchor.get("wikipedia_id")
                or anchor.get("target_wikipedia_id")
                or anchor.get("target_id")
            )
            target_id = _safe_int(value)
            if target_id is not None:
                target_ids.add(target_id)

    return target_ids


def extract_document(record: Dict[str, Any]) -> Optional[KiltDocument]:
    doc_id = _safe_int(
        record.get("wikipedia_id")
        or record.get("id")
    )

    title = (
        record.get("wikipedia_title")
        or record.get("title")
        or ""
    )

    if doc_id is None or not title:
        return None

    paragraphs = extract_paragraphs(record)
    body = " ".join(paragraphs)
    first_paragraph = paragraphs[0] if paragraphs else ""

    out_links = extract_anchor_target_ids(record)
    out_links.discard(doc_id)

    return KiltDocument(
        doc_id=doc_id,
        title=title.strip(),
        body=body,
        first_paragraph=first_paragraph,
        out_links=out_links,
    )


def load_kilt_documents(
    records: Iterable[Dict[str, Any]],
    limit: Optional[int] = None,
) -> Dict[int, KiltDocument]:
    documents: Dict[int, KiltDocument] = {}

    for i, record in enumerate(records):
        if limit is not None and i >= limit:
            break

        doc = extract_document(record)
        if doc is None:
            continue

        documents[doc.doc_id] = doc

    return documents


def build_title_to_id(
    documents: Dict[int, KiltDocument],
) -> Dict[str, int]:
    title_to_id: Dict[str, int] = {}

    for doc_id, doc in documents.items():
        title_to_id[normalize_title(doc.title)] = doc_id

    return title_to_id


def sample_by_bfs(
    documents: Dict[int, KiltDocument],
    target_size: int = 500,
    max_depth: int = 2,
    random_seed: int = 42,
    num_auto_seeds: int = 20,
) -> Dict[int, KiltDocument]:
    """
    Sample a Wikipedia subset using BFS.

    Seeds are automatically selected by internal out-degree among loaded docs.
    This is suitable for streaming mode because fixed title seeds may not appear
    in the loaded prefix.

    Edge direction:
        source document -> linked target document
    """
    random.seed(random_seed)

    internal_out_links = compute_internal_out_links(documents)

    seed_ids = select_seed_ids_by_out_degree(
        internal_out_links=internal_out_links,
        num_seeds=num_auto_seeds,
    )

    if not seed_ids:
        raise ValueError(
            "No valid seed documents were found.\n"
            f"Loaded documents: {len(documents)}\n"
            f"Documents with internal out-links: "
            f"{sum(1 for links in internal_out_links.values() if links)}\n"
            "Try increasing --load-limit or check anchor parsing."
        )

    sampled_ids: Set[int] = set()
    queue: deque[Tuple[int, int]] = deque()

    for seed_id in seed_ids:
        queue.append((seed_id, 0))

    while queue and len(sampled_ids) < target_size:
        current_id, depth = queue.popleft()

        if current_id in sampled_ids:
            continue
        if current_id not in documents:
            continue

        sampled_ids.add(current_id)

        if depth >= max_depth:
            continue

        neighbors = list(internal_out_links.get(current_id, set()))
        neighbors = sorted(neighbors)
        random.shuffle(neighbors)

        for neighbor_id in neighbors:
            if neighbor_id not in sampled_ids:
                queue.append((neighbor_id, depth + 1))

    sampled_docs = {
        doc_id: documents[doc_id]
        for doc_id in sampled_ids
    }

    return filter_internal_links(sampled_docs)


def filter_internal_links(
    documents: Dict[int, KiltDocument],
) -> Dict[int, KiltDocument]:
    """
    Keep only links whose target document is also in the sampled subset.
    """
    sampled_ids = set(documents.keys())
    filtered: Dict[int, KiltDocument] = {}

    for doc_id, doc in documents.items():
        filtered[doc_id] = KiltDocument(
            doc_id=doc.doc_id,
            title=doc.title,
            body=doc.body,
            first_paragraph=doc.first_paragraph,
            out_links={
                target_id
                for target_id in doc.out_links
                if target_id in sampled_ids and target_id != doc_id
            },
        )

    return filtered


def build_documents_for_index(
    documents: Dict[int, KiltDocument],
) -> Dict[int, Dict[str, str]]:
    """
    Convert KILT documents into the format expected by InvertedIndex.add_documents().
    """
    return {
        doc_id: {
            "title": doc.title,
            "body": doc.body,
        }
        for doc_id, doc in documents.items()
    }


def build_edges(
    documents: Dict[int, KiltDocument],
) -> List[Tuple[int, int]]:
    """
    Return graph edges as (source_doc_id, target_doc_id).
    """
    edges: List[Tuple[int, int]] = []

    for source_id, doc in documents.items():
        for target_id in sorted(doc.out_links):
            edges.append((source_id, target_id))

    return edges


def build_queries_and_relevance(
    documents: Dict[int, KiltDocument],
    query_body_chars: int = 200,
    include_self: bool = True,
    include_outlinks: bool = True,
    skip_empty_relevance: bool = True,
) -> Tuple[Dict[int, str], Dict[int, Set[int]]]:
    """
    Build synthetic QRY/REL pairs.

    Query:
        title + first 200 characters of first paragraph

    Relevance:
        {self_doc_id} union out_links_in_sample
    """
    queries: Dict[int, str] = {}
    relevance: Dict[int, Set[int]] = {}

    for doc_id, doc in documents.items():
        rel_docs: Set[int] = set()

        if include_self:
            rel_docs.add(doc_id)

        if include_outlinks:
            rel_docs.update(doc.out_links)

        if skip_empty_relevance and not rel_docs:
            continue

        query_text = doc.title
        if doc.first_paragraph:
            query_text += " " + doc.first_paragraph[:query_body_chars]

        queries[doc_id] = query_text.strip()
        relevance[doc_id] = rel_docs

    return queries, relevance


def build_sample_meta(
    documents: Dict[int, KiltDocument],
    target_size: int,
    max_depth: int,
    num_auto_seeds: int,
) -> Dict[str, Any]:
    return {
        "sampling_method": "auto_seed_bfs_by_internal_out_degree",
        "target_size": target_size,
        "actual_size": len(documents),
        "max_depth": max_depth,
        "num_auto_seeds": num_auto_seeds,
        "sampled_doc_ids": sorted(documents.keys()),
        "titles": {
            doc_id: doc.title
            for doc_id, doc in documents.items()
        },
        "num_edges": len(build_edges(documents)),
    }


def build_kilt_subset(
    records: Iterable[Dict[str, Any]],
    target_size: int = 500,
    max_depth: int = 2,
    load_limit: Optional[int] = None,
    random_seed: int = 42,
    num_auto_seeds: int = 20,
) -> Dict[str, Any]:
    """
    End-to-end helper.

    Returns:
        {
            "documents": documents_for_index,
            "raw_documents": sampled KiltDocument dict,
            "edges": edge list,
            "queries": query_id -> query text,
            "relevance": query_id -> set(doc_ids),
            "meta": sampling metadata
        }
    """
    all_docs = load_kilt_documents(records, limit=load_limit)

    print(f"Loaded docs: {len(all_docs)}")
    print(
        "Docs with raw out-links: "
        f"{sum(1 for d in all_docs.values() if d.out_links)}"
    )
    print(
        "Total raw out-links: "
        f"{sum(len(d.out_links) for d in all_docs.values())}"
    )

    sampled_docs = sample_by_bfs(
        documents=all_docs,
        target_size=target_size,
        max_depth=max_depth,
        random_seed=random_seed,
        num_auto_seeds=num_auto_seeds,
    )

    print(f"Sampled docs: {len(sampled_docs)}")
    print(f"Internal edges: {len(build_edges(sampled_docs))}")

    documents_for_index = build_documents_for_index(sampled_docs)
    edges = build_edges(sampled_docs)
    queries, relevance = build_queries_and_relevance(sampled_docs)
    meta = build_sample_meta(
        documents=sampled_docs,
        target_size=target_size,
        max_depth=max_depth,
        num_auto_seeds=num_auto_seeds,
    )

    return {
        "documents": documents_for_index,
        "raw_documents": sampled_docs,
        "edges": edges,
        "queries": queries,
        "relevance": relevance,
        "meta": meta,
    }

def compute_internal_out_links(
    documents: Dict[int, KiltDocument],
) -> Dict[int, Set[int]]:
    """
    Compute out-links that stay inside the loaded document set.
    """
    loaded_ids = set(documents.keys())

    return {
        doc_id: {
            target_id
            for target_id in doc.out_links
            if target_id in loaded_ids and target_id != doc_id
        }
        for doc_id, doc in documents.items()
    }


def select_seed_ids_by_out_degree(
    internal_out_links: Dict[int, Set[int]],
    num_seeds: int = 20,
) -> List[int]:
    """
    Select seed documents by internal out-degree.
    """
    ranked = sorted(
        internal_out_links.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    return [
        doc_id
        for doc_id, links in ranked[:num_seeds]
        if len(links) > 0
    ]