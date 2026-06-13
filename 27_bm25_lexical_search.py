# Semantic search is great at understanding meaning, but it struggles with exact term matching —
# if a user searches for a specific incident ID like "INC-2023-Q4-011", semantic search might
# return sections that are conceptually related (e.g. financial risk) but don't actually contain
# that ID at all, simply because those sections talk about similar themes.
# BM25 (Best Matching 25) solves this: it's a lexical search algorithm that works purely on
# word frequency — it tokenises the query, counts how often each term appears across all documents,
# and gives higher weight to rare, specific terms while ignoring common words like "the" or "a".
# This means "INC-2023-Q4-011" scores almost zero in documents that don't contain it, and very
# high in documents that do — the exact opposite behaviour of semantic search.
# BM25 returns a distance score where 0.0 means a perfect term match and 1.0 means no match.
# The right production strategy is hybrid search: run both semantic and BM25 in parallel and
# merge results — semantic catches conceptual queries, BM25 catches exact IDs and rare terms.

# Full example: this program loads report.md (which contains three incident IDs), chunks it by
# section, builds a BM25Index from scratch (no external library), searches for a specific incident
# ID that semantic search would struggle with, and prints the distance score for every chunk so you
# can see exactly how BM25 weights rare, specific terms higher than common words.


import math
import os
import re
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

REPORT_PATH = os.path.join(os.path.dirname(__file__), "generated_materials", "report.md")


# ─────────────────────────────────────────────
# STEP 1: Chunking (same as file 26)
# ─────────────────────────────────────────────

def chunk_by_section(text: str) -> list[str]:
    parts = re.split(r"\n## ", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            chunks.append(("## " + part) if i > 0 else part)
    return chunks


# ─────────────────────────────────────────────
# BM25Index — implements the BM25 algorithm from scratch
#
# How BM25 scores a query against a document:
#   score = Σ  IDF(term) × TF_normalised(term, doc)
#
# IDF (Inverse Document Frequency): log((N - df + 0.5) / (df + 0.5) + 1)
#   → rare terms (low df) get high IDF; common terms (high df) get low IDF
#
# TF_normalised: (freq × (k1+1)) / (freq + k1 × (1 - b + b × doc_len/avg_len))
#   → controls how much repeated terms boost the score (k1) and
#     penalises very long documents (b)
# ─────────────────────────────────────────────

K1 = 1.5   # term frequency saturation — higher = more reward for repeated terms
B  = 0.75  # length normalisation — 1.0 = full penalty for long docs, 0.0 = no penalty


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())


class BM25Index:
    def __init__(self):
        self._docs: list[dict]          = []   # raw metadata
        self._term_freqs: list[Counter] = []   # term freq per doc
        self._doc_lengths: list[int]    = []   # token count per doc
        self._df: Counter               = Counter()   # document frequency per term
        self._avg_len: float            = 0.0

    def add_document(self, metadata: dict) -> None:
        tokens = _tokenise(metadata.get("content", ""))
        tf = Counter(tokens)

        self._docs.append(metadata)
        self._term_freqs.append(tf)
        self._doc_lengths.append(len(tokens))

        for term in tf:
            self._df[term] += 1

        total = sum(self._doc_lengths)
        self._avg_len = total / len(self._doc_lengths) if self._doc_lengths else 1.0

    def _idf(self, term: str) -> float:
        N  = len(self._docs)
        df = self._df.get(term, 0)
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _score(self, doc_idx: int, query_tokens: list[str]) -> float:
        tf     = self._term_freqs[doc_idx]
        dl     = self._doc_lengths[doc_idx]
        score  = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf   = self._idf(term)
            tf_n  = (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * dl / self._avg_len))
            score += idf * tf_n
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        """Returns [(metadata, distance), ...] sorted by distance ascending (0 = best match)."""
        query_tokens = _tokenise(query)
        raw_scores   = [self._score(i, query_tokens) for i in range(len(self._docs))]

        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            max_score = 1.0

        # Convert to distance: 0.0 = perfect match, 1.0 = no match at all
        results = [
            (self._docs[i], 1.0 - (raw_scores[i] / max_score))
            for i in range(len(self._docs))
        ]
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def __len__(self):
        return len(self._docs)


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

print("=== BM25 Lexical Search Demo ===\n")

if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(f"report.md not found at {REPORT_PATH}")

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_by_section(text)
print(f"[Step 1] Chunked report.md into {len(chunks)} sections.\n")

# Build BM25 index — no embeddings, no API call, instant
store = BM25Index()
for chunk in chunks:
    store.add_document({"content": chunk})
print(f"[Step 2] Built BM25 index with {len(store)} documents.\n")

print("Ready. Ask questions — especially ones with specific IDs or rare terms.")
print("Try: 'What happened with INC-2023-Q4-011?'     <- BM25 shines here")
print("     'How many engineers were promoted?'        <- works well too")
print("     'Tell me about company performance'        <- semantic would be better here")
print("\nType 'all' to see all distance scores for the last query.")
print("Type 'exit' to quit.\n")

last_all_results: list[tuple[dict, float]] = []

while True:
    query = input("You : ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if query.lower() == "all":
        if not last_all_results:
            print("  No query run yet.\n")
        else:
            print("\n  All chunks scored (last query):")
            for doc, dist in last_all_results:
                header = doc["content"].splitlines()[0][:50]
                bar    = "#" * int((1 - dist) * 30)
                print(f"    dist={dist:.4f}  {header:<50}  {bar}")
            print()
        continue

    if not query:
        continue

    # Search — returns top 3 by default
    top_results = store.search(query, top_k=3)
    last_all_results = store.search(query, top_k=len(store))

    print("\n  Top matches (lower distance = better term match):\n")
    for rank, (doc, dist) in enumerate(top_results, 1):
        header  = doc["content"].splitlines()[0]
        snippet = " ".join(doc["content"].split()[1:30])  # first 30 words after header
        bar     = "#" * int((1 - dist) * 30)
        print(f"  [{rank}] dist={dist:.4f}  {bar}")
        print(f"       {header}")
        print(f"       {snippet}...")
        print()
