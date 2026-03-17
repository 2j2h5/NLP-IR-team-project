import pickle
from pathlib import Path
import kagglehub


DATASET_HANDLE = "dmaso01dsta/cisi-a-dataset-for-information-retrieval"
OUTPUT_DIR = Path("./cisi_index")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================
# 1. 데이터셋 다운로드
# =========================

def download_dataset():
    path = kagglehub.dataset_download(DATASET_HANDLE)
    return Path(path)


# =========================
# 2. CISI.QRY 찾기
# =========================

def find_qry_file(dataset_dir: Path) -> Path:
    candidates = list(dataset_dir.rglob("CISI.QRY"))

    if not candidates:
        raise FileNotFoundError("CISI.QRY 파일을 찾을 수 없습니다.")

    return candidates[0]


# =========================
# 3. CISI.QRY 파싱
# =========================

def parse_cisi_qry(file_path: Path) -> dict[int, str]:
    """
    CISI.QRY 파일을 파싱해서

    {
        query_id: query_text
    }

    형태로 반환
    """

    queries = {}

    current_query_id = None
    collecting_text = False
    text_lines = []

    def save_query():
        nonlocal current_query_id, text_lines

        if current_query_id is None:
            return

        query_text = " ".join(text_lines).strip()
        queries[current_query_id] = query_text

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith(".I "):
                save_query()

                current_query_id = int(line.split()[1])
                collecting_text = False
                text_lines = []

            elif line == ".W":
                collecting_text = True

            elif line.startswith("."):
                collecting_text = False

            else:
                if collecting_text:
                    text_lines.append(line.strip())

    save_query()

    return queries


# =========================
# 4. pickle 저장
# =========================

def save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# =========================
# 5. 메인 실행
# =========================

def main():
    print("[1/3] 데이터셋 다운로드 중...")
    dataset_dir = download_dataset()
    print("다운로드 경로:", dataset_dir)

    print("[2/3] CISI.QRY 찾는 중...")
    qry_path = find_qry_file(dataset_dir)
    print("CISI.QRY:", qry_path)

    print("[3/3] QRY 파싱 중...")
    queries = parse_cisi_qry(qry_path)

    save_pickle(queries, OUTPUT_DIR / "queries.pkl")

    print("\n완료되었습니다.")
    print("쿼리 개수:", len(queries))
    print("저장 파일:", OUTPUT_DIR / "queries.pkl")


if __name__ == "__main__":
    main()