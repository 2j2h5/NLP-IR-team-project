import re
import pickle
from collections import defaultdict
from pathlib import Path

import kagglehub


# =========================
# 1. 설정
# =========================

DATASET_HANDLE = "dmaso01dsta/cisi-a-dataset-for-information-retrieval"
OUTPUT_DIR = Path("./cisi_index")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 2. 전처리 함수
# =========================

def tokenize(text: str) -> list[str]:
    """
    텍스트를 소문자화하고,
    알파벳/숫자가 아닌 문자는 공백으로 바꾼 뒤 토큰화합니다.
    stopword 제거는 하지 않습니다.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = text.split()
    return tokens


# =========================
# 3. 데이터셋 다운로드 및 파일 찾기
# =========================

def download_dataset() -> Path:
    """
    kagglehub로 데이터셋을 다운로드하고
    다운로드된 폴더 경로를 반환합니다.
    """
    path = kagglehub.dataset_download(DATASET_HANDLE)
    return Path(path)


def find_cisi_all(dataset_dir: Path) -> Path:
    """
    다운로드된 폴더 아래에서 CISI.ALL 파일을 찾습니다.
    """
    candidates = list(dataset_dir.rglob("CISI.ALL"))

    if not candidates:
        raise FileNotFoundError(
            f"CISI.ALL 파일을 찾지 못했습니다.\n"
            f"다운로드 경로: {dataset_dir}"
        )

    if len(candidates) > 1:
        print("[경고] CISI.ALL 후보가 여러 개 발견되었습니다. 첫 번째 파일을 사용합니다.")
        for c in candidates:
            print(" -", c)

    return candidates[0]


# =========================
# 4. CISI.ALL 파싱
# =========================

def parse_cisi_all(file_path: Path) -> dict[int, dict]:
    """
    CISI.ALL 파일을 파싱해서
    {doc_id: {"title": ..., "author": ..., "body": ...}}
    형태로 반환합니다.
    """
    documents = {}

    current_doc_id = None
    current_field = None

    title_lines = []
    author_lines = []
    body_lines = []

    def save_current_document():
        nonlocal current_doc_id, title_lines, author_lines, body_lines

        if current_doc_id is None:
            return

        documents[current_doc_id] = {
            "title": " ".join(title_lines).strip(),
            "author": " ".join(author_lines).strip(),
            "body": " ".join(body_lines).strip(),
        }

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith(".I "):
                save_current_document()

                current_doc_id = int(line.split()[1])
                current_field = None
                title_lines = []
                author_lines = []
                body_lines = []

            elif line == ".T":
                current_field = "title"
            elif line == ".A":
                current_field = "author"
            elif line == ".W":
                current_field = "body"
            elif line.startswith("."):
                # 다른 필드(.X 등)는 현재 사용하지 않음
                current_field = None
            else:
                if current_field == "title":
                    title_lines.append(line.strip())
                elif current_field == "author":
                    author_lines.append(line.strip())
                elif current_field == "body":
                    body_lines.append(line.strip())

    save_current_document()
    return documents


# =========================
# 5. 역인덱스 생성
# =========================

def build_inverted_index(documents: dict[int, dict]) -> tuple[dict, dict]:
    """
    역인덱스를 생성합니다.

    index 구조:
    {
        term: {
            doc_id: {
                "title_tf": int,
                "body_tf": int
            }
        }
    }

    doc_token_info 구조:
    {
        doc_id: {
            "title_tokens": [...],
            "body_tokens": [...],
            "title_len": int,
            "body_len": int
        }
    }
    """
    index = defaultdict(dict)
    doc_token_info = {}

    for doc_id, doc in documents.items():
        title_tokens = tokenize(doc["title"])
        body_tokens = tokenize(doc["body"])

        doc_token_info[doc_id] = {
            "title_tokens": title_tokens,
            "body_tokens": body_tokens,
            "title_len": len(title_tokens),
            "body_len": len(body_tokens),
        }

        title_tf_count = defaultdict(int)
        body_tf_count = defaultdict(int)

        for tok in title_tokens:
            title_tf_count[tok] += 1

        for tok in body_tokens:
            body_tf_count[tok] += 1

        all_terms = set(title_tf_count.keys()) | set(body_tf_count.keys())

        for term in all_terms:
            index[term][doc_id] = {
                "title_tf": title_tf_count.get(term, 0),
                "body_tf": body_tf_count.get(term, 0),
            }

    return dict(index), doc_token_info


# =========================
# 6. 단어 가중치 계산
# =========================

def compute_term_weights(index: dict, total_docs: int) -> dict[str, float]:
    """
    사용자님 식:
        weight(term) = (N / df(term)) - 1

    N: 전체 문서 수
    df(term): 그 term이 등장한 문서 수
    """
    term_weights = {}

    for term, posting in index.items():
        df = len(posting)
        weight = (total_docs / df) - 1
        term_weights[term] = weight

    return term_weights


# =========================
# 7. pickle 저장
# =========================

def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =========================
# 8. 메인
# =========================

def main():
    print("[1/5] 데이터셋 다운로드 중...")
    dataset_dir = download_dataset()
    print("다운로드 경로:", dataset_dir)

    print("[2/5] CISI.ALL 파일 찾는 중...")
    cisi_all_path = find_cisi_all(dataset_dir)
    print("CISI.ALL 경로:", cisi_all_path)

    print("[3/5] CISI.ALL 파싱 중...")
    documents = parse_cisi_all(cisi_all_path)
    total_docs = len(documents)
    print(f"총 문서 수: {total_docs}")

    print("[4/5] 역인덱스 생성 중...")
    index, doc_token_info = build_inverted_index(documents)
    print(f"어휘 수(vocabulary size): {len(index)}")

    print("[5/5] 단어 가중치 계산 및 pickle 저장 중...")
    term_weights = compute_term_weights(index, total_docs)

    save_pickle(documents, OUTPUT_DIR / "documents.pkl")
    save_pickle(doc_token_info, OUTPUT_DIR / "doc_token_info.pkl")
    save_pickle(index, OUTPUT_DIR / "inverted_index.pkl")
    save_pickle(term_weights, OUTPUT_DIR / "term_weights.pkl")

    metadata = {
        "dataset_handle": DATASET_HANDLE,
        "dataset_dir": str(dataset_dir),
        "cisi_all_path": str(cisi_all_path),
        "total_docs": total_docs,
        "vocab_size": len(index),
    }
    save_pickle(metadata, OUTPUT_DIR / "metadata.pkl")

    print("\n완료되었습니다.")
    print("저장 폴더:", OUTPUT_DIR.resolve())
    print("생성된 파일:")
    print("- documents.pkl")
    print("- doc_token_info.pkl")
    print("- inverted_index.pkl")
    print("- term_weights.pkl")
    print("- metadata.pkl")


if __name__ == "__main__":
    main()