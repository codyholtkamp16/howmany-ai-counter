"""
Schematic Item Counter — Flask Backend
1. Reads a legend PDF to extract all item codes + descriptions
2. Counts every legend item across all schematic pages
3. Returns annotated PDFs (one per item) + a summary table

Requirements:
    pip install flask flask-cors openai pdf2image pillow pypdf reportlab

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 server.py
"""

import base64
import io
import os
import json
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from openai import OpenAI
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter

try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pypdfium2 as pdfium
    PDFIUM_AVAILABLE = True
except ImportError:
    PDFIUM_AVAILABLE = False

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL   = "gpt-4o"

client = OpenAI(api_key=API_KEY) if API_KEY else None


# ── PDF / image helpers ───────────────────────────────────────────────────────

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 150) -> list[Image.Image]:
    if PDF2IMAGE_AVAILABLE:
        return convert_from_bytes(pdf_bytes, dpi=dpi)
    if PDFIUM_AVAILABLE:
        images = []
        pdf = pdfium.PdfDocument(pdf_bytes)
        for i in range(len(pdf)):
            page   = pdf[i]
            scale  = dpi / 72.0
            bitmap = page.render(scale=scale)
            images.append(bitmap.to_pil())
        return images
    raise RuntimeError("No PDF renderer available. Install poppler-utils or pypdfium2.")


def image_to_base64(img: Image.Image, max_side: int = 2048) -> tuple[str, str]:
    w, h = img.size
    if max(w, h) > max_side:
        ratio = max_side / max(w, h)
        img   = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return b64, "image/png"


# ── Step 1: Parse legend into structured list ─────────────────────────────────

def parse_legend(legend_images: list[Image.Image]) -> list[dict]:
    """
    Returns a list of dicts: [{"code": "EL-01", "description": "String Lighting"}, ...]
    """
    if client is None:
        raise RuntimeError("API key is not set.")

    content = []
    for img in legend_images:
        b64, media_type = image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{b64}", "detail": "high"},
        })

    content.append({
        "type": "text",
        "text": """This is a legend/key from an architectural or site-furnishing schematic.
Read it carefully and extract every single item listed.
Each item has a symbol code (like EL-01, SF-03, PV-04) and a description.

Respond ONLY with valid JSON in this exact format and nothing else:
{
  "items": [
    {"code": "EL-01", "description": "String Lighting"},
    {"code": "EL-02", "description": "Tree Uplight"}
  ]
}"""
    })

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1024,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )

    raw = response.choices[0].message.content.strip()
    print("=== LEGEND PARSED ===")
    print(raw)
    print("=====================")

    try:
        data = json.loads(raw)
        return data.get("items", [])
    except json.JSONDecodeError:
        print("!!! Failed to parse legend JSON")
        return []


# ── Step 2: Count one item across all schematic pages ────────────────────────

def count_item_in_schematics(schematic_images: list[Image.Image],
                              item_code: str,
                              item_description: str,
                              legend_summary: str) -> list[dict]:
    """
    Returns page_results list for one item code.
    """
    if client is None:
        raise RuntimeError("API key is not set.")

    page_results = []

    for i, img in enumerate(schematic_images, start=1):
        orig_w, orig_h = img.size
        b64, media_type = image_to_base64(img)

        max_side = 2048
        if max(orig_w, orig_h) > max_side:
            ratio = max_side / max(orig_w, orig_h)
            img_w = int(orig_w * ratio)
            img_h = int(orig_h * ratio)
        else:
            img_w, img_h = orig_w, orig_h

        prompt = f"""You are an expert at reading architectural and site-furnishing schematics.

LEGEND SUMMARY:
{legend_summary}

TASK:
Find every instance of the following item in this schematic drawing (page {i} of {len(schematic_images)}):

CODE: {item_code}
DESCRIPTION: {item_description}

The image is {img_w} x {img_h} pixels.

Instructions:
- Look for the text label "{item_code}" placed as a callout or tag anywhere in the drawing.
- Do NOT count entries inside the legend/key box itself — only placements in the actual drawing area.
- For each instance found, provide the EXACT CENTER coordinate in pixels (x from left, y from top).
- The coordinate should point precisely to the CENTER of the text label itself, not nearby.
- Be as precise as possible — coordinates will be used to draw a small circle directly on top of the label.
- Classify as "confirmed" (certain) or "maybe" (ambiguous or partially visible).

Respond ONLY with this exact JSON and nothing else:
{{
  "confirmed": [
    {{"x": <int>, "y": <int>}}
  ],
  "maybe": [
    {{"x": <int>, "y": <int>}}
  ],
  "confidence": "<high|medium|low>",
  "reasoning": "<brief explanation>",
  "notes": "<any caveats>"
}}"""

        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=2048,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )

        raw = response.choices[0].message.content.strip()
        print(f"=== {item_code} page {i} ===")
        print(raw)

        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {"confirmed": [], "maybe": [], "confidence": "low",
                      "reasoning": raw, "notes": "Parse failed."}

        result["page"]  = i
        result["img_w"] = img_w
        result["img_h"] = img_h
        page_results.append(result)

    return page_results


# ── Step 3: Annotate PDF for one item ────────────────────────────────────────

def annotate_pdf_for_item(schematic_images: list[Image.Image],
                           page_results: list[dict],
                           dpi: int,
                           item_code: str) -> bytes:
    confirmed_stroke = Color(0.85, 0.1,  0.1,  alpha=0.9)
    confirmed_text   = Color(0.85, 0.1,  0.1,  alpha=1.0)
    maybe_fill       = Color(1.0,  0.6,  0.1,  alpha=0.35)
    maybe_stroke     = Color(0.9,  0.45, 0.05, alpha=0.9)
    maybe_text       = Color(0.7,  0.3,  0.0,  alpha=1.0)

    circle_r  = 5
    box_pad   =  5
    font_size =  5

    writer = PdfWriter()

    for page_idx, img in enumerate(schematic_images):
        result = next((r for r in page_results if r["page"] == page_idx + 1), None)

        rgb_img = img.convert("RGB") if img.mode != "RGB" else img
        img_buf = io.BytesIO()
        rgb_img.save(img_buf, format="PNG")
        img_buf.seek(0)

        img_w, img_h = rgb_img.size
        pts_per_px   = 72.0 / dpi
        pdf_w        = img_w * pts_per_px
        pdf_h        = img_h * pts_per_px

        page_buf = io.BytesIO()
        c = canvas.Canvas(page_buf, pagesize=(pdf_w, pdf_h))
        c.drawImage(ImageReader(img_buf), 0, 0, width=pdf_w, height=pdf_h)

        if result:
            confirmed = result.get("confirmed", [])
            maybe     = result.get("maybe",     [])
            claude_w  = result.get("img_w", img_w)
            claude_h  = result.get("img_h", img_h)
            cx_scale  = img_w / claude_w
            cy_scale  = img_h / claude_h
            item_num  = 1

            for pt in confirmed:
                px = pt.get("x", 0) * cx_scale * pts_per_px
                py = pdf_h - (pt.get("y", 0) * cy_scale * pts_per_px)
                c.setStrokeColor(confirmed_stroke)
                c.setLineWidth(1.5)
                c.circle(px, py, circle_r, stroke=1, fill=0)
                c.setFillColor(confirmed_text)
                c.setFont("Helvetica-Bold", font_size)
                c.drawCentredString(px, py + circle_r + 1, str(item_num))
                item_num += 1

            for pt in maybe:
                px = pt.get("x", 0) * cx_scale * pts_per_px
                py = pdf_h - (pt.get("y", 0) * cy_scale * pts_per_px)
                c.setFillColor(maybe_fill)
                c.setStrokeColor(maybe_stroke)
                c.setLineWidth(1.2)
                c.rect(px - box_pad, py - box_pad,
                       box_pad * 2, box_pad * 2, stroke=1, fill=1)
                c.setFillColor(maybe_text)
                c.setFont("Helvetica-Bold", font_size)
                c.drawCentredString(px, py + box_pad + 1, str(item_num))
                item_num += 1

        c.save()
        page_buf.seek(0)
        writer.add_page(PdfReader(page_buf).pages[0])

    out = io.BytesIO()
    writer.write(out)
    out.seek(0)
    return out.read()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "api_key_set": bool(API_KEY),
        "pdf2image":   PDF2IMAGE_AVAILABLE,
        "pdfium":      PDFIUM_AVAILABLE,
    })


@app.route("/count", methods=["POST"])
def count_items():
    """
    Expects multipart/form-data:
      - legend:   the legend/key PDF (required)
      - files[]:  one or more schematic PDFs (required)
      - dpi:      render DPI (optional, default 150)
    """
    if "legend" not in request.files or not request.files["legend"].filename:
        return jsonify({"error": "No legend PDF uploaded."}), 400
    if "files[]" not in request.files:
        return jsonify({"error": "No schematic PDFs uploaded."}), 400

    dpi = int(request.form.get("dpi", 150))

    try:
        # ── Render legend ──────────────────────────────────────────────────
        legend_bytes  = request.files["legend"].read()
        legend_images = pdf_bytes_to_images(legend_bytes, dpi=dpi)
        legend_items  = parse_legend(legend_images)

        if not legend_items:
            return jsonify({"error": "Could not read any items from the legend."}), 422

        # Build plain-text legend summary for prompts
        legend_summary = "\n".join(
            f"{item['code']}: {item['description']}" for item in legend_items
        )

        # ── Render schematics ──────────────────────────────────────────────
        schematic_images = []
        for pdf_file in request.files.getlist("files[]"):
            if not pdf_file.filename.lower().endswith(".pdf"):
                return jsonify({"error": f"{pdf_file.filename} is not a PDF."}), 400
            schematic_images.extend(pdf_bytes_to_images(pdf_file.read(), dpi=dpi))

        if not schematic_images:
            return jsonify({"error": "Could not render schematic pages."}), 422

        # ── Count every legend item ────────────────────────────────────────
        results_by_item = []

        for item in legend_items:
            code        = item["code"]
            description = item["description"]

            page_results    = count_item_in_schematics(
                schematic_images, code, description, legend_summary
            )
            total_confirmed = sum(len(r.get("confirmed", [])) for r in page_results)
            total_maybe     = sum(len(r.get("maybe",     [])) for r in page_results)

            # Only annotate if at least one instance found
            annotated_b64 = None
            if total_confirmed + total_maybe > 0:
                annotated_bytes = annotate_pdf_for_item(
                    schematic_images, page_results, dpi, code
                )
                annotated_b64 = base64.standard_b64encode(annotated_bytes).decode("utf-8")

            results_by_item.append({
                "code":          code,
                "description":   description,
                "total":         total_confirmed,
                "maybe":         total_maybe,
                "annotated_pdf": annotated_b64,
                "pages": [
                    {
                        "page":       r["page"],
                        "count":      len(r.get("confirmed", [])),
                        "maybe":      len(r.get("maybe",     [])),
                        "confidence": r.get("confidence", ""),
                        "reasoning":  r.get("reasoning",  ""),
                        "notes":      r.get("notes",       ""),
                    }
                    for r in page_results
                ],
            })

        return jsonify({
            "legend_items":    legend_items,
            "results_by_item": results_by_item,
        })

    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("DEBUG", "0") == "1"
    print(f"Starting Schematic Counter server on http://0.0.0.0:{port}")
    if not API_KEY:
        print("  WARNING: OPENAI_API_KEY is not set!")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
