# This file implements the RAG flow as a proper, reusable system rather than a conceptual demo.
# Three things are new compared to file 25: first, the document is read from a real file (report.md)
# instead of hardcoded strings — this is how production systems work, reading from disk or a database.
# Second, generate_embedding() now accepts either a single string or a list of strings, so you can
# embed all chunks in one API call (batch mode) instead of one call per chunk — much faster and cheaper.
# Third, the vector store is wrapped in a VectorIndex class with add_vector() and search() methods,
# making it a self-contained component you could swap out for a real database like Pinecone or Chroma
# without touching any other part of the code.
# search() returns (doc, distance) pairs sorted by cosine distance (1 - similarity), where lower
# distance means higher similarity — 0.0 is a perfect match, 2.0 is completely opposite meaning.
# The clean separation into Chunk → Embed → Store → Query → Search → Generate steps means each
# piece can be improved or replaced independently as your RAG system grows.

# Full example: this program reads report.md, splits it by section headers, batch-embeds all chunks,
# stores them in a VectorIndex, then opens an interactive loop where every question is embedded,
# searched against the store, and the top matching chunks are inserted into a Claude prompt — printing
# the cosine distance scores for every chunk so you can see exactly why certain sections were chosen.


import math
import os
import re
from dotenv import load_dotenv
import voyageai
from anthropic import Anthropic

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
MODEL          = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS     = int(os.getenv("MAX_TOKENS", "1024"))
VOYAGE_MODEL   = "voyage-3-large"
TOP_K          = 2   # number of chunks to retrieve per query

if not VOYAGE_API_KEY or VOYAGE_API_KEY == "your_voyage_api_key_here":
    raise ValueError("VOYAGE_API_KEY not set in .env — sign up at voyageai.com")
if not ANTHROPIC_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

voyage_client    = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)


# ─────────────────────────────────────────────
# STEP 1: Chunking — split a markdown file by ## headers
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
# STEP 2: Embedding — updated to handle both a single string AND a list
# Single string  → returns one list[float]
# List of strings → returns list[list[float]]  (batch mode — one API call, more efficient)
# ─────────────────────────────────────────────

def generate_embedding(text, input_type: str = "document"):
    if isinstance(text, list):
        result = voyage_client.embed(text, model=VOYAGE_MODEL, input_type=input_type)
        return result.embeddings          # list of vectors
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings[0]           # single vector


# ─────────────────────────────────────────────
# STEP 3: VectorIndex — encapsulates storage and search
# Swapping to a real database only requires replacing this class.
# ─────────────────────────────────────────────

def _cosine_distance(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    sim   = dot / (mag_a * mag_b) if mag_a and mag_b else 0.0
    return 1.0 - sim   # distance: 0 = identical, 2 = opposite


class VectorIndex:
    def __init__(self):
        self._store: list[tuple[list[float], dict]] = []

    def add_vector(self, embedding: list[float], metadata: dict) -> None:
        self._store.append((embedding, metadata))

    def search(self, query_embedding: list[float], top_k: int = 2) -> list[tuple[dict, float]]:
        """Returns [(metadata, distance), ...] sorted by distance ascending (lower = more similar)."""
        scored = [
            (meta, _cosine_distance(query_embedding, emb))
            for emb, meta in self._store
        ]
        scored.sort(key=lambda x: x[1])
        return scored[:top_k]

    def __len__(self):
        return len(self._store)


# ─────────────────────────────────────────────
# STEP 6: Generation — build prompt and call Claude
# ─────────────────────────────────────────────

def build_prompt(query: str, results: list[tuple[dict, float]]) -> str:
    context = "\n\n".join(doc["content"] for doc, _ in results)
    return (
        "Answer the user's question using only the information in the report sections below.\n"
        "If the answer is not present, say so clearly.\n\n"
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

print("=== Implementing RAG Flow Demo ===\n")

# Load document
report_path = os.path.join(os.path.dirname(__file__), "generated_materials", "report.md")
if not os.path.exists(report_path):
    raise FileNotFoundError(f"report.md not found at {report_path}")

with open(report_path, "r", encoding="utf-8") as f:
    text = f.read()

# STEP 1 — chunk
chunks = chunk_by_section(text)
print(f"[Step 1] Chunked report.md into {len(chunks)} sections:")
for i, c in enumerate(chunks):
    header = c.splitlines()[0][:60]
    print(f"  chunk[{i}] {header}")

# STEP 2 — batch embed all chunks in one API call
print(f"\n[Step 2] Batch-embedding {len(chunks)} chunks (one API call)...")
embeddings = generate_embedding(chunks, input_type="document")
print(f"  Each embedding has {len(embeddings[0])} dimensions.")

# STEP 3 — build vector index
print(f"\n[Step 3] Building VectorIndex...")
store = VectorIndex()
for embedding, chunk in zip(embeddings, chunks):
    store.add_vector(embedding, {"content": chunk})
print(f"  Indexed {len(store)} chunks.\n")

print("Ready. Ask questions about the annual report.")
print("Tip: 'What did the software engineering dept do?' (tricky — medical section has 'infection vectors')")
print("     'What is the company's financial performance?'")
print("     'How many employees does the company have?'")
print("\nType 'exit' to quit.\n")

while True:
    query = input("You : ").strip()
    if query.lower() == "exit":
        print("Goodbye!")
        break
    if not query:
        continue

    # STEP 4 — embed query
    print("\n[Step 4] Embedding query...")
    query_embedding = generate_embedding(query, input_type="query")

    # STEP 5 — search
    print("[Step 5] Searching vector store...\n")
    results = store.search(query_embedding, top_k=TOP_K)

    print("  Cosine distances (lower = more relevant):")
    all_scored = store.search(query_embedding, top_k=len(store))
    for doc, dist in all_scored:
        header  = doc["content"].splitlines()[0][:45]
        bar     = "#" * int((1 - dist) * 25)
        flag    = " <-- retrieved" if (doc, dist) in results else ""
        print(f"    dist={dist:.4f}  {header:<45}  {bar}{flag}")

    # STEP 6 — generate
    print(f"\n[Step 6] Sending top {TOP_K} chunk(s) to Claude...\n")
    prompt = build_prompt(query, results)
    answer = ask_claude(prompt)

    print(f"Claude : {answer}\n")
