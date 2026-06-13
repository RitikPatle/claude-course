# After chunking your document, the next challenge is finding which chunks are actually relevant
# to a user's question — and the answer is semantic search using text embeddings.
# A text embedding is just a long list of numbers (typically 1024+ floats, each between -1 and +1)
# that represents the *meaning* of a piece of text; similar meanings produce similar number patterns.
# Unlike keyword search that only matches exact words, semantic search compares these number patterns
# so "heart attack" and "myocardial infarction" are understood as the same concept.
# Anthropic doesn't provide an embedding model, so the recommended provider is VoyageAI — you need
# a separate account and API key (free to start) added as VOYAGE_API_KEY in your .env file.
# The input_type parameter matters: use "document" when embedding chunks you want to store, and
# "query" when embedding the user's question — VoyageAI optimises each differently for best results.
# The numbers in an embedding are not human-readable: we know similar texts get similar vectors, but
# the individual dimensions are learned by the model during training and have no direct interpretation.

# Full example: this program generates embeddings for three text chunks and one query using VoyageAI,
# prints the first 10 numbers of each vector so you can see the raw format, then computes cosine
# similarity between the query and each chunk to demonstrate that semantically related text scores
# higher even when it shares no exact keywords — showing exactly why embeddings beat keyword search.


import os
import math
from dotenv import load_dotenv
import voyageai

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
VOYAGE_MODEL = "voyage-3-large"

if not VOYAGE_API_KEY or VOYAGE_API_KEY == "your_voyage_api_key_here":
    raise ValueError(
        "VOYAGE_API_KEY not set in .env file.\n"
        "Sign up at https://www.voyageai.com and add your key to .env"
    )

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)


# ─────────────────────────────────────────────
# Embedding generation
# input_type="document" for chunks you store, "query" for user questions
# ─────────────────────────────────────────────

def generate_embedding(text: str, input_type: str = "query") -> list[float]:
    result = voyage_client.embed([text], model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings[0]


# ─────────────────────────────────────────────
# Cosine similarity — measures how similar two vectors are.
# Score of 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite meaning.
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────
# Sample chunks — same three topics as file 23
# ─────────────────────────────────────────────

CHUNKS = [
    {
        "id": "chunk_1",
        "topic": "Medical Research",
        "text": (
            "Researchers identified a critical bug in the immune response pathway. "
            "The study involved 500 patients over three years. Results showed a 40% "
            "improvement in treatment outcomes when the new protocol was applied."
        ),
    },
    {
        "id": "chunk_2",
        "topic": "Software Engineering",
        "text": (
            "Engineers fixed 127 bugs this quarter, improving system stability by 35%. "
            "The team deployed three major releases and increased code review coverage "
            "to 94% across all active repositories."
        ),
    },
    {
        "id": "chunk_3",
        "topic": "Financial Summary",
        "text": (
            "Revenue grew by 12% year over year. Operating costs decreased due to "
            "infrastructure optimisation. The new pricing model drove a 20% increase "
            "in enterprise subscriptions."
        ),
    },
]

# A query that contains "bugs" — keyword search would return chunk_1 AND chunk_2
# but semantic search should rank chunk_2 (software) much higher.
QUERY = "How many bugs did the engineering team fix this quarter?"


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

print("=== Text Embeddings Demo ===")
print(f"Model : {VOYAGE_MODEL}\n")

# Step 1 — embed the query
print(f'Query : "{QUERY}"\n')
print("Generating query embedding...")
query_embedding = generate_embedding(QUERY, input_type="query")
print(f"  Vector length  : {len(query_embedding)} dimensions")
print(f"  First 10 values: {[round(x, 4) for x in query_embedding[:10]]}")

# Step 2 — embed each chunk and compute similarity
print("\nGenerating chunk embeddings and scoring...\n")
print(f"{'Chunk':<20} {'Topic':<25} {'Similarity':>10}  {'Result'}")
print("-" * 70)

results = []
for chunk in CHUNKS:
    chunk_embedding = generate_embedding(chunk["text"], input_type="document")
    score = cosine_similarity(query_embedding, chunk_embedding)
    results.append((score, chunk))

# Sort highest similarity first
results.sort(key=lambda x: x[0], reverse=True)

for score, chunk in results:
    bar = "#" * int(score * 30)
    flag = " <-- best match" if score == results[0][0] else ""
    print(f"{chunk['id']:<20} {chunk['topic']:<25} {score:>10.4f}  {bar}{flag}")

print()
print("Interpretation:")
print("  Even though 'bugs' appears in chunk_1 (Medical) too,")
print("  semantic search correctly ranks chunk_2 (Software) highest")
print("  because the *meaning* of the query matches Software Engineering.")
print()
print("Next step: store these embeddings in a vector database")
print("  so you can search thousands of chunks instantly (see file 25).")
