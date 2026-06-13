# Citations turn Claude from a "black box" into a transparent research assistant — instead of
# just giving you an answer, Claude also shows exactly which sentence in your document it used
# to arrive at that answer, so users can verify the information themselves.
# Enabling citations requires only two extra fields in your document block: a "title" (a readable
# name for the document) and "citations": {"enabled": True} — everything else stays the same.
# When citations are on, Claude's response is no longer a single text string; it becomes a list
# of interleaved blocks where text blocks carry a "citations" attribute listing every source
# reference made in that sentence — each citation contains the cited_text (the exact quote from
# your document), the document_index (which file it came from), and either page numbers (for PDFs)
# or character positions (for plain text sources).
# Citations work with both PDFs and plain text: PDFs return start_page_number/end_page_number
# while plain text sources return start_char_index/end_char_index for pinpoint accuracy.
# This feature is especially valuable when users need to trust Claude's answers — they can click
# through to the exact passage rather than having to search the whole document themselves.

# Full example: this program demonstrates citations using both a PDF (the company_report.pdf from
# file 31) and a plain text article, sends the same factual question to Claude with citations
# enabled, then renders every response block showing the answer text alongside each citation's
# quoted source text and its location — giving you a complete, production-ready citation renderer.


import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

client = Anthropic(api_key=API_KEY)

GENERATED_DIR = Path(__file__).parent / "generated_materials"
GENERATED_DIR.mkdir(exist_ok=True)
PDF_PATH      = GENERATED_DIR / "company_report.pdf"


# ─────────────────────────────────────────────
# Sample plain-text article for the text-source demo
# ─────────────────────────────────────────────

ARTICLE_TEXT = """
Acme Corp Q3 2025 Performance Brief

The Software Division generated $21.3 million in Q3 revenue, representing 22% year-over-year growth.
Customer churn fell from 5.1% to 3.2%, the lowest level in company history.
Four major product releases shipped during the quarter with zero critical defects in production.

The Medical Division posted $16.8 million in Q3 revenue, up 9% year-over-year.
The XDR-47 antibody protocol received FDA phase-3 clearance in August.
The research team grew from 58 to 71 headcount to support expanded clinical trials.

The Hardware Division contributed $9.1 million in Q3 revenue, a modest 4% increase.
Component shortages in July caused temporary margin pressure that resolved by September.

Total company Q3 revenue reached $47.2 million against a target of $44.0 million.
Full-year guidance has been raised to $176 million from the previous estimate of $170 million.
The Board approved a $5 million increase to the 2026 R&D budget in the October meeting.
""".strip()


# ─────────────────────────────────────────────
# Citation response renderer
# Iterates over every block and prints text + inline citation details.
# ─────────────────────────────────────────────

def render_with_citations(response) -> None:
    citation_counter = 0

    for block in response.content:
        if block.type != "text":
            continue

        print(block.text)

        citations = getattr(block, "citations", []) or []
        for cite in citations:
            citation_counter += 1
            cited_text   = getattr(cite, "cited_text", "").strip()
            doc_title    = getattr(cite, "document_title", "unknown")
            doc_index    = getattr(cite, "document_index", 0)
            cite_type    = getattr(cite, "type", "")

            # Location differs by source type
            if "page" in cite_type:
                start = getattr(cite, "start_page_number", "?")
                end   = getattr(cite, "end_page_number", "?")
                loc   = f"page {start}–{end}"
            elif "char" in cite_type:
                start = getattr(cite, "start_char_index", "?")
                end   = getattr(cite, "end_char_index", "?")
                loc   = f"chars {start}–{end}"
            else:
                loc = "location unknown"

            print(f"\n  [{citation_counter}] Source: \"{doc_title}\"  (doc #{doc_index}, {loc})")
            print(f"       Cited: \"{cited_text[:120]}{'...' if len(cited_text) > 120 else ''}\"")

    if citation_counter == 0:
        print("\n  (No citations returned — try a more specific factual question)")


# ─────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────

def ask_with_pdf_citations(question: str) -> None:
    if not PDF_PATH.exists():
        print(f"  PDF not found at {PDF_PATH} — run file 31 first to generate it.")
        return

    with open(PDF_PATH, "rb") as f:
        file_bytes = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type":       "base64",
                        "media_type": "application/pdf",
                        "data":       file_bytes,
                    },
                    "title":     "company_report.pdf",   # ← new: readable document name
                    "citations": {"enabled": True},       # ← new: turn citations on
                },
                {"type": "text", "text": question},
            ],
        }],
    )
    render_with_citations(response)


def ask_with_text_citations(question: str) -> None:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type":       "text",          # ← plain text, not base64
                        "media_type": "text/plain",
                        "data":       ARTICLE_TEXT,
                    },
                    "title":     "Q3 Performance Brief",
                    "citations": {"enabled": True},
                },
                {"type": "text", "text": question},
            ],
        }],
    )
    render_with_citations(response)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

print("=== Citations Demo ===")
print(f"Model: {MODEL}\n")

print("Choose a demo mode:")
print("  [1] PDF citations   — uses company_report.pdf (file 31)")
print("  [2] Text citations  — uses an in-memory plain-text article")
print("  [3] Both            — same question sent to both sources, side by side")
print("  [4] Interactive     — choose source and ask your own questions")
print()
mode = input("Mode (1/2/3/4, default 3): ").strip() or "3"

SAMPLE_QUESTION = "What were the key financial results for each division this quarter?"


if mode == "1":
    print(f'\nQuestion: "{SAMPLE_QUESTION}"\n')
    print("─" * 55 + "  PDF WITH CITATIONS  " + "─" * 10)
    ask_with_pdf_citations(SAMPLE_QUESTION)

elif mode == "2":
    print(f'\nQuestion: "{SAMPLE_QUESTION}"\n')
    print("─" * 55 + "  PLAIN TEXT WITH CITATIONS  " + "─" * 5)
    ask_with_text_citations(SAMPLE_QUESTION)

elif mode == "3":
    print(f'\nQuestion: "{SAMPLE_QUESTION}"\n')

    print("=" * 60)
    print("  SOURCE: PDF (company_report.pdf) — citations show page numbers")
    print("=" * 60)
    ask_with_pdf_citations(SAMPLE_QUESTION)

    print("\n" + "=" * 60)
    print("  SOURCE: Plain text — citations show character positions")
    print("=" * 60)
    ask_with_text_citations(SAMPLE_QUESTION)

elif mode == "4":
    source = input("Source — [1] PDF  [2] Plain text (default 2): ").strip() or "2"
    print("\nType your questions. Type 'exit' to quit.\n")
    while True:
        question = input("You : ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        print()
        if source == "1":
            ask_with_pdf_citations(question)
        else:
            ask_with_text_citations(question)
        print()

else:
    print("Invalid mode.")
