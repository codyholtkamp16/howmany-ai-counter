"""
Schematic Item Counter — Flask Backend
Converts uploaded PDF pages to images and sends them to the Anthropic Vision API
for intelligent item counting on site furnishing schematics/drawings.

Requirements:
    pip install flask flask-cors anthropic pdf2image pillow pypdf

Usage:
    ANTHROPIC_API_KEY=sk-... python server.py
"""

import base64
import io
import os
import json
import tempfile
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import anthropic
from PIL import Image

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    print("WARNING: pdf2image not available. Install poppler-utils for PDF rendering.")

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Allow requests from the frontend

# ── Anthropic client ──────────────────────────────────────────────────────────
API_KEY = "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"
client = anthropic.Anthropic(api_key=API_KEY) if API_KEY else None


# ── Helpers ───────────────────────────────────────────────────────────────────

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    """Render every page of a PDF to a PIL Image."""
    images = []

    if PDF2IMAGE_AVAILABLE:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        images.extend(imgs)
    elif PDFIUM_AVAILABLE:
        pdf = pdfium.PdfDocument(pdf_bytes)
        for i in range(len(pdf)):
            page = pdf[i]
            scale = dpi / 72.0
            bitmap = page.render(scale=scale)
            pil_img = bitmap.to_pil()
            images.append(pil_img)
    else:
        raise RuntimeError(
            "No PDF renderer available. "
            "Install poppler-utils (for pdf2image) or pypdfium2."
        )

    return images


def image_to_base64(img: Image.Image, max_side: int = 2048) -> tuple[str, str]:
    """Resize if needed and return (base64_string, media_type)."""
    # Cap resolution to keep token usage reasonable
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

    # Convert to RGB (strip alpha if present)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


def build_counting_prompt(item_description: str, page_num: int, total_pages: int) -> str:
    page_info = f"(page {page_num} of {total_pages})" if total_pages > 1 else ""
    return f"""You are an expert at reading architectural and site-furnishing schematics/blueprints.

I need you to count every instance of the following item in this schematic drawing {page_info}:

ITEM TO COUNT: {item_description}

Instructions:
1. Carefully examine the entire drawing, including legends, callouts, and plan views.
2. Count EVERY occurrence — both in the main drawing and in any detail views or legends.
3. If the same item appears in a legend/key, do NOT count the legend entry itself — only count real placements.
4. Provide your reasoning step-by-step, then give a final definitive count.

Respond in this exact JSON format (and nothing else):
{{
  "count": <integer>,
  "confidence": "<high|medium|low>",
  "reasoning": "<brief explanation of where you found them>",
  "notes": "<any caveats or ambiguities>"
}}"""


def call_claude_vision(
    b64_images: list[tuple[str, str]], item_description: str
) -> dict:
    """Send all page images to Claude and aggregate counts."""
    if client is None:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")

    page_results = []
    total_count = 0

    for i, (b64, media_type) in enumerate(b64_images, start=1):
        prompt = build_counting_prompt(item_description, i, len(b64_images))

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        raw = response.content[0].text.strip()

        # Strip potential markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "count": 0,
                "confidence": "low",
                "reasoning": raw,
                "notes": "Could not parse structured response.",
            }

        result["page"] = i
        page_results.append(result)
        total_count += result.get("count", 0)

    return {
        "total_count": total_count,
        "pages": page_results,
        "item": item_description,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "api_key_set": bool(API_KEY),
        "pdf2image": PDF2IMAGE_AVAILABLE,
        "pdfium": PDFIUM_AVAILABLE,
    })


@app.route("/count", methods=["POST"])
def count_items():
    """
    Expects multipart/form-data with:
      - file: the PDF file
      - item: string description of the item to count
      - dpi: (optional) render DPI, default 150
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    if "item" not in request.form or not request.form["item"].strip():
        return jsonify({"error": "No item description provided."}), 400

    pdf_file = request.files["file"]
    item_description = request.form["item"].strip()
    dpi = int(request.form.get("dpi", 150))

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Uploaded file must be a PDF."}), 400

    try:
        pdf_bytes = pdf_file.read()

        # Render PDF pages to images
        images = pdf_bytes_to_images(pdf_bytes, dpi=dpi)

        if not images:
            return jsonify({"error": "Could not render any pages from the PDF."}), 422

        # Convert images to base64
        b64_images = [image_to_base64(img) for img in images]

        # Call Claude Vision
        result = call_claude_vision(b64_images, item_description)

        return jsonify(result)

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("DEBUG", "1") == "1"
    print(f"Starting Schematic Counter server on http://localhost:{port}")
    if not API_KEY:
        print("  ⚠  WARNING: ANTHROPIC_API_KEY is not set!")
    app.run(host="0.0.0.0", port=port, debug=debug)
