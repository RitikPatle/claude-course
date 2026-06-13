# Text chunking is the first step in any RAG pipeline — before Claude can search your documents,
# you must break them into small pieces that fit inside a prompt without overwhelming it.
# A bad chunking strategy is the most common reason RAG systems give wrong answers: if a chunk
# mixes two unrelated topics (e.g. medical research and software bugs in the same block), Claude
# will return irrelevant context that looks plausible but answers the wrong question entirely.
# There are four main strategies: size-based (split every N characters — simple, always works,
# but may cut words mid-sentence), structure-based (split on Markdown headers — clean sections,
# but only works when you control the document format), sentence-based (split on punctuation then
# group sentences — good middle ground for most text), and semantic-based (use NLP to group
# related sentences — best quality but computationally expensive and complex to implement).
# Overlap is the key tuning knob for size-based and sentence-based chunking: each chunk re-uses
# a few characters or sentences from the previous chunk so that context is never lost at boundaries.
# Size-based with overlap is the production default because it is reliable with any document type,
# including code, PDFs, and plain text that has no structural markers whatsoever.

# Full example: this program implements all four strategies, runs them on the same sample document,
# prints every chunk with its index and character count so you can compare quality side by side,
# and shows exactly how overlap prevents context loss at chunk boundaries — giving you a working
# reference you can drop into any RAG pipeline and tune for your specific documents.


import re


# ─────────────────────────────────────────────
# Sample document — intentionally mixes two topics so you can see
# how each strategy keeps or breaks topical boundaries.
# ─────────────────────────────────────────────

SAMPLE_DOCUMENT = """
## Medical Research Update

Researchers have identified a critical bug in the immune response pathway.
The study involved 500 patients over three years. Results showed a 40% improvement
in treatment outcomes when the new protocol was applied. Side effects were minimal
and no adverse reactions were recorded during the trial period.

## Software Engineering Report

Engineers fixed 127 bugs this quarter, improving system stability by 35%.
The team deployed three major releases and onboarded two new frameworks.
Code review coverage increased to 94% across all active repositories.
Automated testing caught 89% of regressions before they reached production.

## Financial Summary

Revenue grew by 12% year over year. Operating costs decreased due to infrastructure
optimisation. The new pricing model drove a 20% increase in enterprise subscriptions.
Cash flow remained positive throughout all four quarters of the fiscal year.
""".strip()


# ─────────────────────────────────────────────
# Strategy 1: Size-based chunking
# Split every chunk_size characters with chunk_overlap overlap.
# Most reliable — works with any document type including code and PDFs.
# ─────────────────────────────────────────────

def chunk_by_char(text: str, chunk_size: int = 150, chunk_overlap: int = 20) -> list[str]:
    chunks = []
    start_idx = 0

    while start_idx < len(text):
        end_idx = min(start_idx + chunk_size, len(text))
        chunks.append(text[start_idx:end_idx])
        start_idx = end_idx - chunk_overlap if end_idx < len(text) else len(text)

    return chunks


# ─────────────────────────────────────────────
# Strategy 2: Structure-based chunking
# Split on Markdown section headers (## ...).
# Cleanest chunks — but only works when you control the document format.
# ─────────────────────────────────────────────

def chunk_by_section(text: str, header_pattern: str = r"\n## ") -> list[str]:
    raw = re.split(header_pattern, text)
    # Re-attach the "## " prefix that the split consumed so chunks are self-contained
    chunks = []
    for i, part in enumerate(raw):
        part = part.strip()
        if not part:
            continue
        # First split piece has no header stripped; rest do
        chunks.append(("## " + part) if i > 0 else part)
    return chunks


# ─────────────────────────────────────────────
# Strategy 3: Sentence-based chunking
# Split into sentences first, then group max_sentences_per_chunk together
# with overlap_sentences re-used from the previous chunk.
# Good middle ground — respects sentence boundaries without needing document structure.
# ─────────────────────────────────────────────

def chunk_by_sentence(
    text: str,
    max_sentences_per_chunk: int = 3,
    overlap_sentences: int = 1,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    start_idx = 0

    while start_idx < len(sentences):
        end_idx = min(start_idx + max_sentences_per_chunk, len(sentences))
        chunks.append(" ".join(sentences[start_idx:end_idx]))
        start_idx += max_sentences_per_chunk - overlap_sentences
        if start_idx < 0:
            start_idx = 0

    return chunks


# ─────────────────────────────────────────────
# Strategy 4: Semantic-based chunking (simplified)
# Full semantic chunking uses sentence embeddings + cosine similarity to detect
# topic shifts. That requires a vector model (e.g. sentence-transformers).
# This simplified version groups sentences until a keyword boundary is detected —
# it is NOT production-quality but illustrates the concept without extra dependencies.
# ─────────────────────────────────────────────

def chunk_by_semantic_simple(text: str, boundary_keywords: list[str] | None = None) -> list[str]:
    if boundary_keywords is None:
        # Default: treat lines starting with ## as topic shifts
        boundary_keywords = ["##"]

    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        is_boundary = any(line.strip().startswith(kw) for kw in boundary_keywords)
        if is_boundary and current:
            chunks.append("\n".join(current).strip())
            current = []
        current.append(line)

    if current:
        chunks.append("\n".join(current).strip())

    return [c for c in chunks if c]


# ─────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────

def print_chunks(strategy_name: str, chunks: list[str]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {strategy_name}  ({len(chunks)} chunks)")
    print(f"{'=' * 60}")
    for i, chunk in enumerate(chunks):
        print(f"\n  [chunk {i + 1}]  {len(chunk)} chars")
        print("  " + "\n  ".join(chunk.splitlines()))


# ─────────────────────────────────────────────
# Main — run all four strategies on the same document and compare
# ─────────────────────────────────────────────

print("=== Text Chunking Strategies Demo ===")
print(f"Document length: {len(SAMPLE_DOCUMENT)} characters\n")
print("The document intentionally mixes Medical, Software, and Financial sections.")
print("Watch how each strategy handles (or blurs) those topic boundaries.\n")

print_chunks(
    "Strategy 1 — Size-based (chunk_size=150, overlap=20)",
    chunk_by_char(SAMPLE_DOCUMENT, chunk_size=150, chunk_overlap=20),
)

print_chunks(
    "Strategy 2 — Structure-based (split on ## headers)",
    chunk_by_section(SAMPLE_DOCUMENT),
)

print_chunks(
    "Strategy 3 — Sentence-based (3 sentences/chunk, 1 overlap)",
    chunk_by_sentence(SAMPLE_DOCUMENT, max_sentences_per_chunk=3, overlap_sentences=1),
)

print_chunks(
    "Strategy 4 — Semantic-based simplified (boundary on ## keywords)",
    chunk_by_semantic_simple(SAMPLE_DOCUMENT),
)

print(f"\n{'=' * 60}")
print("Summary:")
print("  Size-based    -> always works, may cut sentences mid-way")
print("  Structure     -> cleanest chunks, needs formatted documents")
print("  Sentence      -> respects sentence boundaries, good general purpose")
print("  Semantic      -> best quality, needs NLP/embeddings in production")
print(f"{'=' * 60}\n")
