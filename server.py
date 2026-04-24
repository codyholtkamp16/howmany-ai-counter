"""
Schematic Item Counter — Flask Backend
1. Reads a legend PDF to extract all item codes + descriptions
2. For each legend item, tiles each schematic page into a 2x2 grid
   and counts instances in each tile separately (4x API calls per page)
   then maps coordinates back to the full page for annotation.
3. Returns annotated PDFs (one per item) + a summary table.

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

# Grid dimensions — 2x2 = 4 tiles per page
TILE_COLS = 2
TILE_ROWS = 2

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


def tile_image(img: Image.Image, cols: int, rows: int) -> list[dict]:
    """
    Split image into a cols x rows grid of tiles.
    Returns list of dicts with keys:
      tile_img, x_offset, y_offset, tile_w, tile_h
    """
    full_w, full_h = img.size
    tile_w = full_w // cols
    tile_h = full_h // rows
    tiles  = []

    for row in range(rows):
        for col in range(cols):
            x0 = col * tile_w
            y0 = row * tile_h
            # Extend last tile to cover any remainder pixels
            x1 = full_w if col == cols - 1 else x0 + tile_w
            y1 = full_h if row == rows - 1 else y0 + tile_h
            tile_img = img.crop((x0, y0, x1, y1))
            tiles.append({
                "tile_img":  tile_img,
                "x_offset":  x0,
                "y_offset":  y0,
                "tile_w":    x1 - x0,
                "tile_h":    y1 - y0,
            })

    return tiles


# ── Step 1: Parse legend ──────────────────────────────────────────────────────

def parse_legend(legend_images: list[Image.Image]) -> list[dict]:
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


# ── Step 2: Count one item using tiled scanning ───────────────────────────────

def count_tile(tile_img: Image.Image,
               item_code: str,
               item_description: str,
               legend_summary: str,
               page_num: int,
               total_pages: int,
               tile_idx: int,
               total_tiles: int) -> dict:
    """
    Send one tile to GPT-4o and return confirmed/maybe lists
    in tile-local pixel coordinates.
    """
    orig_w, orig_h = tile_img.size
    b64, media_type = image_to_base64(tile_img)

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
You are looking at tile {tile_idx + 1} of {total_tiles} of page {page_num} of {total_pages}.
Find every instance of the following item in THIS TILE ONLY:

CODE: {item_code}
DESCRIPTION: {item_description}

This tile image is {img_w} x {img_h} pixels.

Instructions:
- Look for the exact text label "{item_code}" placed as a callout or tag in the drawing.
- Do NOT count entries inside a legend/key box — only real placements in the drawing.
- For each instance found, provide its EXACT CENTER coordinate in pixels
  (x from left edge of this tile, y from top edge of this tile).
- Be as precise as possible — coordinates are used to draw a small circle on the label.
- Classify as "confirmed" (certain) or "maybe" (ambiguous/partially visible).
- If you find nothing, return empty arrays.

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
    print(f"  Tile {tile_idx + 1}/{total_tiles}: {raw[:120]}...")

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"confirmed": [], "maybe": [], "confidence": "low",
                  "reasoning": raw, "notes": "Parse failed."}

    result["img_w"] = img_w
    result["img_h"] = img_h
    result["tile_orig_w"] = orig_w
    result["tile_orig_h"] = orig_h
    return result


def count_item_in_schematics(schematic_images: list[Image.Image],
                              item_code: str,
                              item_description: str,
                              legend_summary: str) -> list[dict]:
    """
    Tiles each schematic page into a TILE_COLS x TILE_ROWS grid,
    queries GPT-4o on each tile, then merges coordinates back into
    full-page space. Returns one result dict per page.
    """
    if client is None:
        raise RuntimeError("API key is not set.")

    page_results = []
    total_tiles  = TILE_COLS * TILE_ROWS

    for page_idx, img in enumerate(schematic_images):
        page_num = page_idx + 1
        full_w, full_h = img.size

        print(f"=== {item_code} — page {page_num}/{len(schematic_images)} "
              f"({full_w}x{full_h}px, {total_tiles} tiles) ===")

        tiles = tile_image(img, TILE_COLS, TILE_ROWS)

        all_confirmed = []
        all_maybe     = []
        confidences   = []
        reasonings    = []

        for tile_idx, tile_info in enumerate(tiles):
            tile_result = count_tile(
                tile_info["tile_img"],
                item_code, item_description, legend_summary,
                page_num, len(schematic_images),
                tile_idx, total_tiles,
            )

            tile_orig_w = tile_info["tile_w"]
            tile_orig_h = tile_info["tile_h"]
            img_w       = tile_result.get("img_w", tile_orig_w)
            img_h       = tile_result.get("img_h", tile_orig_h)

            # Scale from (possibly downscaled) tile coords → full page coords
            scale_x = tile_orig_w / img_w
            scale_y = tile_orig_h / img_h

            def to_full(pt):
                return {
                    "x": int(pt["x"] * scale_x) + tile_info["x_offset"],
                    "y": int(pt["y"] * scale_y) + tile_info["y_offset"],
                }

            all_confirmed.extend(to_full(p) for p in tile_result.get("confirmed", []))
            all_maybe.extend(    to_full(p) for p in tile_result.get("maybe",     []))
            confidences.append(tile_result.get("confidence", "low"))
            if tile_result.get("reasoning"):
                reasonings.append(f"Tile {tile_idx+1}: {tile_result['reasoning']}")

        # Overall confidence = worst across tiles
        if "low"    in confidences: overall_conf = "low"
        elif "medium" in confidences: overall_conf = "medium"
        else:                         overall_conf = "high"

        page_results.append({
            "page":       page_num,
            "confirmed":  all_confirmed,
            "maybe":      all_maybe,
            "img_w":      full_w,
            "img_h":      full_h,
            "confidence": overall_conf,
            "reasoning":  " | ".join(reasonings),
            "notes":      "",
        })

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
    box_pad   = 5
    font_size = 5

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
            # Coordinates are already in full-page pixel space
            item_num  = 1

            for pt in confirmed:
                px = pt["x"] * pts_per_px
                py = pdf_h - (pt["y"] * pts_per_px)
                c.setStrokeColor(confirmed_stroke)
                c.setLineWidth(1.2)
                c.circle(px, py, circle_r, stroke=1, fill=0)
                c.setFillColor(confirmed_text)
                c.setFont("Helvetica-Bold", font_size)
                c.drawCentredString(px, py + circle_r + 1, str(item_num))
                item_num += 1

            for pt in maybe:
                px = pt["x"] * pts_per_px
                py = pdf_h - (pt["y"] * pts_per_px)
                c.setFillColor(maybe_fill)
                c.setStrokeColor(maybe_stroke)
                c.setLineWidth(1.0)
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
        "tile_grid":   f"{TILE_COLS}x{TILE_ROWS}",
    })


@app.route("/count", methods=["POST"])
def count_items():
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

        # ── Count every legend item (tiled) ───────────────────────────────
        results_by_item = []

        for item in legend_items:
            code        = item["code"]
            description = item["description"]

            page_results    = count_item_in_schematics(
                schematic_images, code, description, legend_summary
            )
            total_confirmed = sum(len(r.get("confirmed", [])) for r in page_results)
            total_maybe     = sum(len(r.get("maybe",     [])) for r in page_results)

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
