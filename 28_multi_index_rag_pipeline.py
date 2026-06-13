# Files 26 and 27 built two separate search indexes — VectorIndex (semantic) and BM25Index (lexical).
# This file combines them into a single Retriever class that queries both simultaneously.
# The challenge is that the two indexes use completely different scoring systems (cosine distance
# vs BM25 normalised distance), so you cannot simply average or concatenate their scores.
# Reciprocal Rank Fusion (RRF) solves this: instead of comparing raw scores, it compares RANKS.
# For each result from each index you compute 1 / (k + rank), where k is a smoothing constant
# (usually 60; lower k makes top ranks matter more), then sum those values across all indexes.
# A document that ranks #1 in both indexes gets a very high RRF score; one that ranks #1 in only
# one index scores lower; one ranked near the bottom of both indexes scores lowest of all.
# Because all indexes share the same add_document() and search() interface (the SearchIndex
# protocol), adding a third or fourth search method in the future requires zero changes to Retriever.

# Full example: this program builds a Retriever over the same report.md used in files 26-27,
# indexes it in both VectorIndex and BM25Index, then runs queries showing the per-index ranks,
# the RRF fusion calculation, and the final merged ranking — demonstrating that the hybrid result
# handles both semantic questions and exact-ID lookups better than either index alone.


import math
import os
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from typing import Any
from dotenv import load_dotenv
import voyageai
from anthropic import Anthropic

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
MODEL          = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1024"))
VOYAGE_MODEL   = "voyage-3-large"
REPORT_PATH    = os.path.join(os.path.dirname(__file__), "generated_materials", "report.md")

if not VOYAGE_API_KEY or VOYAGE_API_KEY == "your_voyage_api_key_here":
    raise ValueError("VOYAGE_API_KEY not set in .env — sign up at voyageai.com")
if not ANTHROPIC_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

voyage_client    = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)


# ─────────────────────────────────────────────
# SearchIndex protocol
# Any class that implements add_document() and search() can be used in Retriever.
# ─────────────────────────────────────────────

class SearchIndex(ABC):
    @abstractmethod
    def add_document(self, metadata: dict[str, Any]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[tuple[dict, float]]: ...

    @abstractmethod
    def __len__(self) -> int: ...


# ─────────────────────────────────────────────
# VectorIndex (semantic search — from file 26)
# ─────────────────────────────────────────────

def _generate_embedding(text, input_type: str = "document"):
    if isinstance(text, list):
        return voyage_client.embed(text, model=VOYAGE_MODEL, input_type=input_type).embeddings
    return voyage_client.embed([text], model=VOYAGE_MODEL, input_type=input_type).embeddings[0]


def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    sim   = dot / (mag_a * mag_b) if mag_a and mag_b else 0.0
    return 1.0 - sim


class VectorIndex(SearchIndex):
    def __init__(self):
        self._store: list[tuple[list[float], dict]] = []

    def add_document(self, metadata: dict) -> None:
        embedding = _generate_embedding(metadata["content"], input_type="document")
        self._store.append((embedding, metadata))

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        q_emb   = _generate_embedding(query, input_type="query")
        scored  = [(meta, _cosine_distance(q_emb, emb)) for emb, meta in self._store]
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def __len__(self): return len(self._store)


# ─────────────────────────────────────────────
# BM25Index (lexical search — from file 27)
# ─────────────────────────────────────────────

K1, B = 1.5, 0.75

def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())


class BM25Index(SearchIndex):
    def __init__(self):
        self._docs: list[dict]          = []
        self._term_freqs: list[Counter] = []
        self._doc_lengths: list[int]    = []
        self._df: Counter               = Counter()
        self._avg_len: float            = 0.0

    def add_document(self, metadata: dict) -> None:
        tokens = _tokenise(metadata.get("content", ""))
        tf     = Counter(tokens)
        self._docs.append(metadata)
        self._term_freqs.append(tf)
        self._doc_lengths.append(len(tokens))
        for term in tf:
            self._df[term] += 1
        self._avg_len = sum(self._doc_lengths) / len(self._doc_lengths)

    def _score(self, doc_idx: int, query_tokens: list[str]) -> float:
        tf, dl, score = self._term_freqs[doc_idx], self._doc_lengths[doc_idx], 0.0
        N = len(self._docs)
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf   = math.log((N - self._df[term] + 0.5) / (self._df[term] + 0.5) + 1)
            tf_n  = (freq * (K1 + 1)) / (freq + K1 * (1 - B + B * dl / self._avg_len))
            score += idf * tf_n
        return score

    def search(self, query: str, top_k: int = 3) -> list[tuple[dict, float]]:
        tokens     = _tokenise(query)
        raw_scores = [self._score(i, tokens) for i in range(len(self._docs))]
        max_score  = max(raw_scores) if raw_scores else 1.0
        if max_score == 0: max_score = 1.0
        results = [(self._docs[i], 1.0 - raw_scores[i] / max_score) for i in range(len(self._docs))]
        results.sort(key=lambda x: x[1])
        return results[:top_k]

    def __len__(self): return len(self._docs)


# ─────────────────────────────────────────────
# Retriever — wraps multiple indexes, applies RRF to merge results
#
# RRF formula:  score(doc) = Σ  1 / (k_rrf + rank_in_index_i)
#
# k_rrf smooths the effect of top ranks. Lower k = top rank matters more.
# We use k=1 here so results are easier to understand in the demo.
# Production systems typically use k=60.
# ─────────────────────────────────────────────

class Retriever:
    def __init__(self, *indexes: SearchIndex):
        if not indexes:
            raise ValueError("Provide at least one SearchIndex.")
        self._indexes = list(indexes)

    def add_document(self, document: dict[str, Any]) -> None:
        for index in self._indexes:
            index.add_document(document)

    def search(self, query: str, top_k: int = 3, k_rrf: int = 1) -> list[tuple[dict, float]]:
        # Gather ranked results from every index
        all_ranked: list[list[tuple[dict, float]]] = [
            index.search(query, top_k=len(index)) for index in self._indexes
        ]

        # Build RRF score table: doc_id → cumulative RRF score
        # Use id(doc) as a proxy for document identity
        doc_map:   dict[int, dict]  = {}
        rrf_scores: dict[int, float] = defaultdict(float)

        for ranked_list in all_ranked:
            for rank, (doc, _) in enumerate(ranked_list, start=1):
                key = id(doc)
                doc_map[key] = doc
                rrf_scores[key] += 1.0 / (k_rrf + rank)

        # Sort by RRF score descending
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        return [(doc_map[k], rrf_scores[k]) for k in sorted_keys[:top_k]]


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def chunk_by_section(text: str) -> list[str]:
    parts = re.split(r"\n## ", text)
    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            chunks.append(("## " + part) if i > 0 else part)
    return chunks


def build_prompt(query: str, results: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(doc["content"] for doc, _ in results)
    return (
        "Answer the user's question using only the report sections below.\n"
        "If the answer is not present, say so.\n\n"
        f"<user_question>\n{query}\n</user_question>\n\n"
        f"<report>\n{context}\n</report>"
    )


def ask_claude(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

print("=== Multi-Index RAG Pipeline Demo ===\n")

if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(f"report.md not found at {REPORT_PATH}")

with open(REPORT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chunks = chunk_by_section(text)
print(f"[Setup] Chunked report.md into {len(chunks)} sections.")

# Build both indexes via the Retriever (one add_document call populates both)
vector_idx = VectorIndex()
bm25_idx   = BM25Index()
retriever  = Retriever(vector_idx, bm25_idx)

print(f"[Setup] Batch-embedding {len(chunks)} chunks for VectorIndex...")
# Batch embed for efficiency, then add to vector store manually to avoid N API calls
embeddings = _generate_embedding(chunks, input_type="document")
for emb, chunk in zip(embeddings, chunks):
    vector_idx._store.append((emb, {"content": chunk}))
    bm25_idx.add_document({"content": chunk})

print(f"[Setup] Indexed {len(vector_idx)} chunks in VectorIndex, {len(bm25_idx)} in BM25Index.\n")

print("Hybrid search active: VectorIndex (semantic) + BM25Index (lexical) fused via RRF.")
print()
print("Try: 'What happened with INC-2023-Q4-011?'  <- BM25 shines, semantic might drift")
print("     'How did the company perform financially?'  <- semantic shines")
print("     'Tell me about employee satisfaction'  <- semantic shines")
print("\nType 'exit' to quit.\n")

while True:
    query = input("You : ").strip()
    if query.lower() == "exit":
        print("Goodbye!")
        break
    if not query:
        continue

    # Show per-index rankings first
    vec_results = vector_idx.search(query, top_k=3)
    bm25_results = bm25_idx.search(query, top_k=3)

    print("\n  VectorIndex top 3 (semantic):")
    for rank, (doc, dist) in enumerate(vec_results, 1):
        print(f"    #{rank}  dist={dist:.4f}  {doc['content'].splitlines()[0][:55]}")

    print("\n  BM25Index top 3 (lexical):")
    for rank, (doc, dist) in enumerate(bm25_results, 1):
        print(f"    #{rank}  dist={dist:.4f}  {doc['content'].splitlines()[0][:55]}")

    # Fused RRF results
    fused = retriever.search(query, top_k=3, k_rrf=1)
    print("\n  Fused (RRF) top 3:")
    for rank, (doc, rrf_score) in enumerate(fused, 1):
        print(f"    #{rank}  rrf={rrf_score:.4f}  {doc['content'].splitlines()[0][:55]}")

    # Send top 2 fused chunks to Claude
    print(f"\n  Sending top 2 fused chunks to Claude...\n")
    prompt = build_prompt(query, fused[:2])
    answer = ask_claude(prompt)
    print(f"Claude : {answer}\n")
