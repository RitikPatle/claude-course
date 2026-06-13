# Claude's vision capability lets you include images directly in user messages alongside text —
# you encode the image as a base64 string (or pass a URL) and place it in an image block,
# which sits in the same content list as your text block for that message turn.
# The same prompting rules that improve text responses apply to images: a vague prompt like
# "describe this image" gives vague results, while a structured prompt that tells Claude
# exactly what to look for, how to analyse it, and what format to respond in gives accurate,
# reliable results — this is especially important for tasks like object counting or risk scoring
# where a simple question can be surprisingly wrong.
# Key limits to remember: max 100 images per request, 5 MB per image, 8000px max for a single
# image (2000px when sending multiple), and token cost is (width × height) / 750 per image.
# You can also pass images by URL instead of base64 — useful when images are already hosted
# and you want to avoid the overhead of reading and encoding files in your code.

# Full example: this program generates a colourful test image in generated_materials/, then sends
# it to Claude three ways — a simple one-line prompt, a structured step-by-step prompt, and a
# URL-based image — so you can directly compare how prompt quality and delivery method affect
# the quality of Claude's visual analysis response.


import base64
import os
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

API_KEY    = os.getenv("ANTHROPIC_API_KEY")
# Vision works on haiku-4-5, sonnet-4-x, etc. — any current Claude model
MODEL      = os.getenv("MODEL", "claude-haiku-4-5")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

if not API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not set in .env")

client = Anthropic(api_key=API_KEY)

GENERATED_DIR = Path(__file__).parent / "generated_materials"
GENERATED_DIR.mkdir(exist_ok=True)
TEST_IMAGE_PATH = GENERATED_DIR / "test_image.png"


# ─────────────────────────────────────────────
# Generate a sample test image
# Creates a scene with coloured shapes and labels so Claude has something to analyse.
# ─────────────────────────────────────────────

def create_test_image(path: Path) -> None:
    img    = Image.new("RGB", (600, 400), color=(240, 240, 240))
    draw   = ImageDraw.Draw(img)

    # Background sky and ground
    draw.rectangle([0, 0, 600, 250],   fill=(135, 206, 235))   # sky blue
    draw.rectangle([0, 250, 600, 400], fill=(34, 139, 34))     # grass green

    # Sun
    draw.ellipse([480, 30, 560, 110], fill=(255, 220, 0), outline=(255, 165, 0), width=3)

    # House body
    draw.rectangle([180, 160, 380, 280], fill=(210, 105, 30), outline=(101, 67, 33), width=2)
    # Roof
    draw.polygon([(160, 165), (280, 80), (400, 165)], fill=(139, 0, 0), outline=(101, 0, 0))
    # Door
    draw.rectangle([255, 220, 305, 280], fill=(101, 67, 33))
    # Windows
    draw.rectangle([195, 175, 240, 215], fill=(173, 216, 230), outline=(0, 0, 0))
    draw.rectangle([320, 175, 365, 215], fill=(173, 216, 230), outline=(0, 0, 0))

    # Trees
    for x in [70, 440, 510]:
        draw.rectangle([x - 8, 200, x + 8, 260], fill=(101, 67, 33))        # trunk
        draw.ellipse([x - 35, 140, x + 35, 220], fill=(0, 100, 0))          # canopy

    # Clouds
    for cx, cy in [(100, 60), (300, 40)]:
        for dx, dy in [(0, 0), (30, -10), (60, 0), (15, -20), (45, -20)]:
            draw.ellipse([cx + dx - 25, cy + dy - 15, cx + dx + 25, cy + dy + 15],
                         fill=(255, 255, 255))

    # Labels
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, 5),   "Scene Analysis Test Image", fill=(0, 0, 0), font=font)
    draw.text((10, 370), "3 trees  |  1 house  |  1 sun  |  2 clouds", fill=(255, 255, 255), font=font)

    img.save(path, "PNG")
    print(f"  Test image saved: {path}  ({path.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def load_image_base64(path: Path) -> tuple[str, str]:
    """Returns (base64_data, media_type)."""
    suffix_map = {".png": "image/png", ".jpg": "image/jpeg",
                  ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = suffix_map.get(path.suffix.lower(), "image/png")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def estimate_tokens(image_path: Path) -> int:
    with Image.open(image_path) as img:
        w, h = img.size
    tokens = (w * h) // 750
    print(f"  Image size: {w}×{h}px  →  ~{tokens} tokens")
    return tokens


def image_block_base64(path: Path) -> dict:
    data, media_type = load_image_base64(path)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


def image_block_url(url: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "url", "url": url},
    }


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


# ─────────────────────────────────────────────
# Claude caller
# ─────────────────────────────────────────────

def ask_claude(content: list) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

print("=== Image Support Demo ===")
print(f"Model: {MODEL}\n")

# Generate test image
print("Generating test image...")
create_test_image(TEST_IMAGE_PATH)
estimate_tokens(TEST_IMAGE_PATH)

print("\nChoose a demo mode:")
print("  [1] Simple prompt   vs   Structured prompt  (same image, compare quality)")
print("  [2] URL-based image  (pass a public image URL instead of a file)")
print("  [3] Interactive — supply your own image path and prompt")
print()
mode = input("Mode (1/2/3, default 1): ").strip() or "1"


# ── Mode 1: Simple vs Structured ─────────────────────────────────────────────
if mode == "1":
    print("\n" + "=" * 55)
    print("SIMPLE PROMPT — 'What do you see in this image?'")
    print("=" * 55)
    simple_answer = ask_claude([
        image_block_base64(TEST_IMAGE_PATH),
        text_block("What do you see in this image?"),
    ])
    print(simple_answer)

    print("\n" + "=" * 55)
    print("STRUCTURED PROMPT — step-by-step counting methodology")
    print("=" * 55)
    structured_prompt = (
        "Analyse this image using the following steps:\n\n"
        "1. Sky conditions: Describe the sky (colour, weather, any objects).\n"
        "2. Vegetation count: Count every tree visible. Assign each a number as you spot it.\n"
        "3. Structures: Identify any buildings or structures and describe them briefly.\n"
        "4. Verify your tree count using a second pass, scanning left to right.\n\n"
        "Respond with one sentence per step, then a final summary line."
    )
    structured_answer = ask_claude([
        image_block_base64(TEST_IMAGE_PATH),
        text_block(structured_prompt),
    ])
    print(structured_answer)


# ── Mode 2: URL-based image ───────────────────────────────────────────────────
elif mode == "2":
    url = input("\nPaste a public image URL: ").strip()
    if not url.startswith("http"):
        print("URL must start with http/https.")
    else:
        prompt = input("Your question about the image: ").strip() or "Describe what you see."
        print("\nSending URL-based image to Claude...\n")
        answer = ask_claude([
            image_block_url(url),
            text_block(prompt),
        ])
        print(f"Claude : {answer}")


# ── Mode 3: Interactive with own image ────────────────────────────────────────
elif mode == "3":
    print(f"\n(Leave blank to use the generated test image: {TEST_IMAGE_PATH})")
    image_path_str = input("Image file path: ").strip()
    image_path = Path(image_path_str) if image_path_str else TEST_IMAGE_PATH

    if not image_path.exists():
        print(f"File not found: {image_path}")
    else:
        estimate_tokens(image_path)
        print()
        print("Enter your prompt (or press Enter for a default structured prompt):")
        user_prompt = input("> ").strip()
        if not user_prompt:
            user_prompt = (
                "Describe everything you see in this image step by step:\n"
                "1. Main subject\n2. Background\n3. Colours\n4. Any text or labels\n"
                "5. Overall impression in one sentence."
            )
        print("\nSending to Claude...\n")
        answer = ask_claude([
            image_block_base64(image_path),
            text_block(user_prompt),
        ])
        print(f"Claude : {answer}")

else:
    print("Invalid mode.")
