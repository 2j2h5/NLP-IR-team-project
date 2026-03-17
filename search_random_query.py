import pickle
import random
import re
import time
from collections import defaultdict


TOP_N = 3
TITLE_WEIGHT = 2


# =========================
# tokenize
# =========================

def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


# =========================
# pickle 로드
# =========================

with open("cisi_index/inverted_index.pkl", "rb") as f:
    inverted_index = pickle.load(f)

with open("cisi_index/term_weights.pkl", "rb") as f:
    term_weights = pickle.load(f)

with open("cisi_index/documents.pkl", "rb") as f:
    documents = pickle.load(f)

with open("cisi_index/queries.pkl", "rb") as f:
    queries = pickle.load(f)


# =========================
# 랜덤 query 선택
# =========================

qid = random.choice(list(queries.keys()))
query_text = queries[qid]

print("\n==============================")
print("Random Query ID:", qid)
print("Query:", query_text)
print("==============================\n")


# =========================
# 검색 시간 측정 시작
# =========================

search_start_time = time.perf_counter()


# =========================
# query tokenize
# =========================

query_terms = tokenize(query_text)

print("Processed Query Terms:", query_terms)
print()


# =========================
# scoring + term contribution
# =========================

scores = defaultdict(float)

# doc_id -> {term: contribution_score}
term_contributions = defaultdict(lambda: defaultdict(float))

for term in query_terms:
    if term not in inverted_index:
        continue

    postings = inverted_index[term]
    weight = term_weights.get(term, 0.0)

    for doc_id, tf_info in postings.items():
        tf = tf_info["title_tf"] * TITLE_WEIGHT + tf_info["body_tf"]
        contribution = tf * weight

        scores[doc_id] += contribution
        term_contributions[doc_id][term] += contribution


# =========================
# ranking
# =========================

ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================
# 검색 시간 측정 종료
# =========================

search_end_time = time.perf_counter()
search_elapsed = search_end_time - search_start_time


# =========================
# 출력
# =========================

print(f"Search Time: {search_elapsed:.6f} seconds")
print(f"Top {TOP_N} Results\n")

for rank, (doc_id, score) in enumerate(ranked_docs[:TOP_N], start=1):
    doc = documents[doc_id]

    print(f"Rank {rank}")
    print(f"Doc ID: {doc_id}")
    print(f"Score: {score:.4f}")
    print("Title:", doc["title"])
    print()

    print("Term Contributions:")
    contribution_items = sorted(
        term_contributions[doc_id].items(),
        key=lambda x: x[1],
        reverse=True
    )

    if contribution_items:
        for term, contribution in contribution_items:
            print(f"  {term:<15} : {contribution:.4f}")
    else:
        print("  (No contributing query terms found)")

    print()
    print("Body:")
    print(doc["body"])
    print("\n---------------------------------------\n")