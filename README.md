# Schematic Item Counter

AI-powered tool that counts items (trees, benches, bollards, etc.) in site-furnishing
schematics/blueprints by sending rendered PDF pages to the Claude Vision API.

---

## Project Structure

```
schematic-counter/
├── server.py     — Flask backend (PDF → image → Claude Vision → count)
├── index.html    — Frontend (drag-and-drop upload, results display)
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install flask flask-cors anthropic pdf2image pillow pypdf

# On Ubuntu/Debian — needed by pdf2image:
sudo apt-get install poppler-utils

# On macOS:
brew install poppler

# Alternatively, install pypdfium2 as a zero-system-dep renderer:
pip install pypdfium2
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the server

```bash
python server.py
# Listening on http://localhost:5050
```

### 4. Open the frontend

Open `index.html` in your browser directly (double-click or drag into browser).
Make sure the "Server URL" field in the app points to `http://localhost:5050`.

---

## Usage

1. Upload a schematic PDF (multi-page supported).
2. Type (or click a suggestion pill for) the item to count — e.g. *street trees*, *bollards*.
3. Pick a render quality (higher = better accuracy, slower).
4. Click **Analyse schematic**.

The server will:
- Convert each PDF page to a PNG at the chosen DPI.
- Send each image to `claude-opus-4-5` with a structured prompt.
- Parse the JSON response and aggregate counts across all pages.

Results show the total count, per-page breakdowns, confidence level, and Claude's reasoning.

---

## API Reference

### `GET /health`

Returns server status.

```json
{ "status": "ok", "api_key_set": true, "pdf2image": true, "pdfium": false }
```

### `POST /count`

**Form data:**
| Field | Type   | Required | Default | Description                      |
|-------|--------|----------|---------|----------------------------------|
| file  | file   | ✓        | —       | PDF file                         |
| item  | string | ✓        | —       | Natural-language item description|
| dpi   | int    | —        | 150     | Render resolution                |

**Response:**
```json
{
  "total_count": 14,
  "item": "street trees",
  "pages": [
    {
      "page": 1,
      "count": 9,
      "confidence": "high",
      "reasoning": "Nine tree symbols (circles with cross) distributed along...",
      "notes": ""
    },
    {
      "page": 2,
      "count": 5,
      "confidence": "medium",
      "reasoning": "Five planting symbols in the east courtyard...",
      "notes": "Two symbols are partially obscured by dimension lines."
    }
  ]
}
```

---

## Tips for best results

- **Use 200+ DPI** for dense or complex schematics.
- Be specific: *"deciduous street trees (round canopy symbol)"* is better than just *"trees"*.
- If counts seem off, describe the symbol used in the drawing.
- For very large drawings, crop to the relevant area before exporting to PDF.

---

## Environment variables

| Variable          | Default     | Description              |
|-------------------|-------------|--------------------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key   |
| `PORT`            | `5050`      | Server port              |
| `DEBUG`           | `1`         | Flask debug mode (0/1)   |
