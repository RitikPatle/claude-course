# PDF support works almost identically to image support from file 30 — the only differences
# are three small changes in the message block: type becomes "document" instead of "image",
# media_type becomes "application/pdf" instead of "image/png", and the variable is conventionally
# named file_bytes instead of image_bytes to make the code easier to read.
# Claude can extract far more from a PDF than just raw text — it understands embedded images
# and charts, reads tables and preserves their data relationships, recognises document structure
# like headings and sections, and can answer questions about any of these in one API call.
# There is no need to run a separate PDF-to-text converter before sending to Claude; you hand
# it the raw PDF bytes and it handles all parsing internally, making it a one-stop solution
# for any document processing task — summarisation, data extraction, Q&A, or translation.
# PDFs are sent as base64-encoded strings inside a "document" block, exactly like images are
# sent inside an "image" block, so if you already know file 30 the only thing to learn here
# is the three-word swap: document / application/pdf / file_bytes.

# Full example: this program generates a realistic multi-section company report as a PDF using
# only Python's built-in string formatting (no external library), then sends it to Claude with
# four different prompts — summarise, extract the table, answer a specific question, and list
# key numbers — so you can see how Claude handles text, structure, and data from the same file.


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

GENERATED_DIR   = Path(__file__).parent / "generated_materials"
GENERATED_DIR.mkdir(exist_ok=True)
TEST_PDF_PATH   = GENERATED_DIR / "company_report.pdf"


# ─────────────────────────────────────────────
# Minimal pure-Python PDF writer
# PDF is a plain-text format for simple documents — no library needed.
# ─────────────────────────────────────────────

def _pdf_stream(lines: list[str]) -> str:
    """Build a PDF content stream from a list of text commands."""
    return "\n".join(lines)


def create_test_pdf(path: Path) -> None:
    """Create a realistic multi-section PDF without any external library."""

    # PDF content stream — BT/ET = Begin/End Text, Tf = font, Td = move, Tj = show text
    def line(x, y, text, size=11):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F1 {size} Tf {x} {y} Td ({safe}) Tj ET"

    def bold(x, y, text, size=13):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        return f"BT /F2 {size} Tf {x} {y} Td ({safe}) Tj ET"

    content_lines = [
        bold(72, 750, "Acme Corp - Q3 2025 Internal Report", 16),
        bold(72, 720, "1. Executive Summary", 13),
        line(72, 700, "Acme Corp delivered strong Q3 results with revenue of $47.2 million,"),
        line(72, 685, "exceeding the Q3 target of $44.0 million by 7.3%. Net profit reached"),
        line(72, 670, "$8.1 million, up from $6.4 million in Q3 2024. All three divisions"),
        line(72, 655, "contributed positively, with Software leading growth at 22% YoY."),
        bold(72, 625, "2. Division Performance", 13),
        line(72, 605, "Software Division: Revenue $21.3M (+22% YoY). Shipped 4 major product"),
        line(72, 590, "releases. Customer churn reduced to 3.2%, down from 5.1% last quarter."),
        line(72, 575, "Medical Division: Revenue $16.8M (+9% YoY). Phase-3 trial for XDR-47"),
        line(72, 560, "protocol received FDA clearance. Headcount grew from 58 to 71 researchers."),
        line(72, 545, "Hardware Division: Revenue $9.1M (+4% YoY). Supply chain delays in July"),
        line(72, 530, "impacted margins; recovered to normal by September."),
        bold(72, 500, "3. Quarterly Revenue Table (USD millions)", 13),
        line(72, 480, "Division        Q1      Q2      Q3      YTD"),
        line(72, 465, "Software        17.2    19.5    21.3    58.0"),
        line(72, 450, "Medical         14.1    15.9    16.8    46.8"),
        line(72, 435, "Hardware         8.4     8.8     9.1    26.3"),
        line(72, 420, "TOTAL           39.7    44.2    47.2   131.1"),
        bold(72, 390, "4. Key Risks", 13),
        line(72, 370, "- Ongoing component shortages may affect Hardware Division in Q4."),
        line(72, 355, "- Three enterprise contracts (combined $4.2M ARR) up for renewal in Oct."),
        line(72, 340, "- Regulatory review of XDR-47 in the EU expected to conclude in Dec."),
        bold(72, 310, "5. Outlook", 13),
        line(72, 290, "Full-year revenue guidance raised to $176M (previously $170M). Q4 focus:"),
        line(72, 275, "close enterprise renewals, launch Software v4.0, and finalise EU filing."),
        line(72, 260, "Board has approved a $5M R&D budget increase effective January 2026."),
    ]

    stream_content = "\n".join(content_lines)
    stream_bytes   = stream_content.encode("latin-1")
    stream_len     = len(stream_bytes)

    pdf = f"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj

2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj

3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
   /Contents 4 0 R
   /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>
endobj

4 0 obj
<< /Length {stream_len} >>
stream
{stream_content}
endstream
endobj

5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj

6 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>
endobj

xref
0 7
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000{stream_len + 320:06d} 00000 n
0000000{stream_len + 395:06d} 00000 n

trailer
<< /Size 7 /Root 1 0 R >>
startxref
{stream_len + 470}
%%EOF"""

    path.write_bytes(pdf.encode("latin-1"))
    print(f"  Test PDF saved: {path}  ({path.stat().st_size:,} bytes)")


# ─────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────

def load_pdf_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def ask_claude_with_pdf(pdf_base64: str, question: str) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",          # ← "document" not "image"
                    "source": {
                        "type":       "base64",
                        "media_type": "application/pdf",   # ← not "image/png"
                        "data":       pdf_base64,          # ← file_bytes not image_bytes
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

print("=== PDF Support Demo ===")
print(f"Model: {MODEL}\n")

# Create test PDF
print("Generating test PDF (company_report.pdf)...")
create_test_pdf(TEST_PDF_PATH)

# Load once, reuse across all questions
print("Encoding PDF as base64...\n")
pdf_b64 = load_pdf_base64(TEST_PDF_PATH)

print("Choose a demo mode:")
print("  [1] Run four preset questions to show different extraction capabilities")
print("  [2] Interactive — ask your own questions about the PDF")
print()
mode = input("Mode (1/2, default 1): ").strip() or "1"


if mode == "1":
    questions = [
        ("Summarise",        "Summarise this document in two sentences."),
        ("Extract table",    "Extract the revenue table from section 3 and present it clearly."),
        ("Specific fact",    "What was the Software Division's Q3 revenue, and how does it compare to Q1?"),
        ("Key numbers",      "List every financial figure mentioned in the document as bullet points."),
    ]

    for label, question in questions:
        print("=" * 55)
        print(f"  [{label}]  {question}")
        print("=" * 55)
        answer = ask_claude_with_pdf(pdf_b64, question)
        print(answer)
        print()

elif mode == "2":
    print("Ask any question about company_report.pdf. Type 'exit' to quit.\n")
    while True:
        question = input("You : ").strip()
        if question.lower() == "exit":
            print("Goodbye!")
            break
        if not question:
            continue
        answer = ask_claude_with_pdf(pdf_b64, question)
        print(f"\nClaude : {answer}\n")

else:
    print("Invalid mode.")
