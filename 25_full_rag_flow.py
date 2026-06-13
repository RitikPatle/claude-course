# The full RAG pipeline connects all the pieces from files 23 and 24 into one end-to-end system.
# Step 1 is chunking: break the source document into sections (file 23).
# Step 2 is embedding: convert each chunk into a vector of numbers using VoyageAI (file 24).
# Step 3 is storing: keep those chunk + vector pairs in an in-memory "vector store" (a list here;
# a real app would use a dedicated database like Pinecone, Chroma, or pgvector).
# Steps 1-3 are preprocessing — they happen once ahead of time before any user asks anything.
# Step 4 is query embedding: when a user asks a question, embed it with input_type="query".
# Step 5 is retrieval: compute cosine similarity between the query vector and every stored chunk
# vector, then pick the top-N most similar chunks — this is the semantic search step.
# Step 6 is generation: insert those retrieved chunks into a prompt as context and send it to
# Claude; Claude reads the context and answers based on what was actually in your documents.
# The whole point of RAG is that Claude never had to memorise your documents — it reads the
# relevant parts fresh each time, which means it can answer questions about private or recent
# data that was never in its training set.

# Full example: this program embeds two intentionally tricky sections (one medical, one software —
# both containing misleading keywords), indexes them in an in-memory store, then lets you ask
# questions in a loop; for every query it prints the similarity scores for all chunks, highlights
# which one was retrieved, builds the final prompt, and shows Claude's answer — giving you a
# transparent, step-by-step view of the complete RAG pipeline in action.


import math
import os
from dotenv import load_dotenv
import voyageai
from anthropic import Anthropic

load_dotenv()

VOYAGE_API_KEY  = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_KEY   = os.getenv("ANTHROPIC_API_KEY")
MODEL           = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "1024"))
VOYAGE_MODEL    = "voyage-3-large"
TOP_K           = 1   # how many chunks to retrieve per query

if not VOYAGE_API_KEY or VOYAGE_API_KEY == "your_voyage_api_key_here":
    raise ValueError("VOYAGE_API_KEY not set in .env — sign up at voyageai.com")
if not ANTHROPIC_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

voyage_client    = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_KEY)


# ─────────────────────────────────────────────
# Source document — two sections intentionally designed to confuse keyword search.
# "bug" appears in the medical section; "infection vectors" appears in the software section.
# Keyword search would mix them up; semantic search should not.
# ─────────────────────────────────────────────

DOCUMENTS = [
    {
        "id": "section_1",
        "title": "Medical Research",
        "text": (
            "This year saw significant strides in our understanding of XDR-47, a 'bug' "
            "we have not seen before. Clinical trials across 12 hospitals showed a 67% "
            "reduction in patient recovery time using the new antibody protocol. "
            "The research team published three peer-reviewed papers and secured FDA approval "
            "for phase-3 trials beginning next quarter."
        ),
    },
    {
        "id": "section_2",
        "title": "Software Engineering",
        "text": (
            "This division dedicated significant effort to studying various infection vectors "
            "in our distributed systems. Engineers resolved 214 critical defects and deployed "
            "11 production releases with zero downtime. Automated test coverage reached 96% "
            "and the team migrated the core API to a microservices architecture, cutting "
            "average response time by 40%."
        ),
    },
]


# ─────────────────────────────────────────────
# Utility functions (same as file 24)
# ─────────────────────────────────────────────

def generate_embedding(text: str, input_type: str = "query") -> list[float]:
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


# ─────────────────────────────────────────────
# STEP 1-3: Preprocessing — chunk, embed, store
# In a real app this runs once when you ingest documents, not on every query.
# ─────────────────────────────────────────────

def build_vector_store(documents: list[dict]) -> list[dict]:
    print("Building vector store (embedding all chunks)...")
    store = []
    for doc in documents:
        embedding = generate_embedding(doc["text"], input_type="document")
        store.append({**doc, "embedding": embedding})
        print(f"  Embedded: [{doc['id']}] {doc['title']}")
    print(f"  Done — {len(store)} chunks indexed.\n")
    return store


# ─────────────────────────────────────────────
# STEP 4-5: Retrieval — embed query, find top-K similar chunks
# ─────────────────────────────────────────────

def retrieve(query: str, store: list[dict], top_k: int = 1) -> list[dict]:
    query_embedding = generate_embedding(query, input_type="query")

    scored = []
    for entry in store:
        sim  = cosine_similarity(query_embedding, entry["embedding"])
        dist = 1 - sim   # cosine distance: 0 = identical, 2 = opposite
        scored.append({**entry, "similarity": sim, "distance": dist})

    scored.sort(key=lambda x: x["similarity"], reverse=True)

    # Print scores for all chunks so you can see the ranking
    print("  Similarity scores:")
    for item in scored:
        bar   = "#" * int(item["similarity"] * 25)
        flag  = " <-- retrieved" if item in scored[:top_k] else ""
        print(f"    [{item['id']}] {item['title']:<25} sim={item['similarity']:.4f}  dist={item['distance']:.4f}  {bar}{flag}")

    return scored[:top_k]


# ─────────────────────────────────────────────
# STEP 6: Generation — build prompt and send to Claude
# ─────────────────────────────────────────────

def build_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    context_blocks = "\n\n".join(
        f"<section id=\"{c['id']}\" title=\"{c['title']}\">\n{c['text']}\n</section>"
        for c in retrieved_chunks
    )
    return (
        f"Answer the user's question using only the information in the sections below.\n"
        f"If the answer is not in the sections, say so.\n\n"
        f"<user_question>\n{query}\n</user_question>\n\n"
        f"<context>\n{context_blocks}\n</context>"
    )


def ask_claude(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# Main — preprocessing once, then interactive query loop
# ─────────────────────────────────────────────

print("=== Full RAG Pipeline Demo ===\n")

# Preprocessing (steps 1-3) — runs once at startup
vector_store = build_vector_store(DOCUMENTS)

print("Ready. Ask questions about the company report.")
print("The system retrieves the most relevant section before asking Claude.\n")
print("Try: 'What did the software engineering team accomplish this year?'")
print("  or 'Tell me about the medical research results'")
print("  or 'How many bugs were fixed?'  (tricky — 'bug' appears in medical section!)")
print("\nType 'exit' to quit.\n")

while True:
    query = input("You : ").strip()
    if query.lower() == "exit":
        print("Goodbye!")
        break
    if not query:
        continue

    print(f"\n[Step 4] Embedding query...")
    print(f"[Step 5] Searching vector store...")
    top_chunks = retrieve(query, vector_store, top_k=TOP_K)

    print(f"\n[Step 6] Building prompt with retrieved context...")
    prompt = build_prompt(query, top_chunks)

    print(f"[Step 6] Sending to Claude ({MODEL})...\n")
    answer = ask_claude(prompt)

    print(f"Claude : {answer}\n")
