# ir/datasets/cisi.py

from typing import Dict, Set


def parse_cisi_all(file_path: str) -> Dict[int, Dict[str, str]]:
    """
    Parse CISI.ALL document file.

    Returns:
        {
            doc_id: {
                "title": title,
                "body": body
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

            elif (
                line.startswith(".A")
                or line.startswith(".B")
                or line.startswith(".X")
            ):
                current_section = None

            else:
                if current_section == "title":
                    title_lines.append(line.strip())
                elif current_section == "body":
                    body_lines.append(line.strip())

    save_current_document()
    return documents


def parse_cisi_queries(file_path: str) -> Dict[int, str]:
    """
    Parse CISI.QRY query file.

    Returns:
        {
            query_id: query_text
        }
    """
    queries: Dict[int, str] = {}

    current_query_id = None
    current_section = None
    query_lines = []

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

            elif (
                line.startswith(".T")
                or line.startswith(".A")
                or line.startswith(".B")
                or line.startswith(".X")
            ):
                current_section = None

            else:
                if current_section == "body":
                    query_lines.append(line.strip())

    save_current_query()
    return queries


def parse_cisi_rel(file_path: str) -> Dict[int, Set[int]]:
    """
    Parse CISI.REL relevance file.

    Returns:
        {
            query_id: {relevant_doc_id_1, relevant_doc_id_2, ...}
        }
    """
    relevance: Dict[int, Set[int]] = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            query_id = int(parts[0])
            doc_id = int(parts[1])

            if query_id not in relevance:
                relevance[query_id] = set()

            relevance[query_id].add(doc_id)

    return relevance