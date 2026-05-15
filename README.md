# Schematic Item Counter

AI-powered tool that reads a legend/key PDF, extracts every listed symbol code,
then counts those items across one or more schematic PDFs. The backend renders PDF
pages to images, scans each schematic page in tiles with OpenAI vision, and returns
per-item totals plus annotated PDFs.

---

## Project Structure

```text
howmany/
├── Dockerfile       - Recipe for building the app into a Docker image
├── compose.yaml     - Repeatable local/EC2 container run settings
├── requirements.txt - Python dependency list installed inside Docker
├── server.py        - Flask backend: PDF rendering, legend parsing, tiled counting, annotation
├── index.html       - Frontend: legend upload, schematic upload, results table
└── README.md
```

---

## Docker Mental Model

Your current deploy path is:

```text
local files -> GitHub repository -> EC2 pulls repository -> EC2 installs/runs app
```

Docker changes the repeatability part:

```text
project files + Dockerfile -> Docker image -> Docker container running on EC2
```

The terms matter:

- **Dockerfile**: the recipe. It says which Python version to use, which Linux
  packages to install, which Python packages to install, which files to copy, and
  which command starts the app.
- **Image**: the built artifact from that recipe. It is like a frozen snapshot of
  the app plus its runtime dependencies.
- **Container**: a running instance of the image. If the image is the blueprint,
  the container is the live process.
- **Port mapping**: Docker keeps the app inside a private container network, so
  `5050:5050` means "send traffic from EC2/local port 5050 into container port
  5050."
- **Environment variables**: runtime settings such as `OPENAI_API_KEY`. These
  should be passed into the container when it runs, not baked into the image.

You can still use GitHub with Docker, but EC2 no longer needs to understand your
Python setup manually. It just needs Docker, the project files, and the command
to build/run the image. Later, you can push built images to a registry instead of
copying source files to EC2.

---

## Setup Without Docker

### 1. Install dependencies

```bash
pip install flask flask-cors openai pdf2image pillow pypdf reportlab

# On Ubuntu/Debian, needed by pdf2image:
sudo apt-get install poppler-utils

# On macOS:
brew install poppler

# Optional zero-system-dependency PDF renderer fallback:
pip install pypdfium2
```

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

### 3. Start the server

```bash
python server.py
# Listening on http://localhost:5050
```

---

## Setup With Docker

### 1. Build the image

```bash
docker build -t howmany .
```

What this does:

- Reads `Dockerfile`.
- Starts from `python:3.11-slim`.
- Installs Poppler so `pdf2image` can render PDFs.
- Installs Python dependencies from `requirements.txt`.
- Copies `server.py` and `index.html` into the image.
- Produces an image named `howmany`.

### 2. Run the container

```bash
docker run --rm \
  --name howmany \
  -p 5050:5050 \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  howmany
```

What this does:

- `--rm` removes the container after it stops.
- `--name howmany` gives the running container a friendly name.
- `-p 5050:5050` exposes the Flask app to `http://localhost:5050`.
- `-e OPENAI_API_KEY=...` passes your API key at runtime.
- `howmany` is the image to run.

### 3. Or use Docker Compose

```bash
docker compose up --build
```

Compose reads `compose.yaml`, builds the image if needed, maps port `5050`, and
passes `OPENAI_API_KEY` from your shell.

To run it in the background:

```bash
docker compose up --build -d
```

To view logs:

```bash
docker compose logs -f
```

To stop it:

```bash
docker compose down
```

### 4. Open the frontend

Visit `http://localhost:5050` in your browser. If running on EC2, use:

```text
http://YOUR_EC2_PUBLIC_IP:5050
```

Make sure your EC2 security group allows inbound TCP traffic on port `5050`, or
put the app behind a reverse proxy on ports `80`/`443` for a more production-like
deployment.

---

## Usage

1. Upload a legend/key PDF that lists symbol codes and descriptions.
2. Upload one or more schematic PDFs.
3. Pick a render quality. Higher DPI usually improves accuracy but increases
   latency and cost.
4. Click **Count all legend items**.

The server will:

- Convert the legend and schematic PDFs to PNG images.
- Ask the model to extract all legend items as `{code, description}` pairs.
- Split each schematic page into a `3x3` tile grid.
- Count each legend item in each tile.
- Merge tile-local coordinates back into full-page coordinates.
- Generate an annotated PDF for each item with confirmed and possible matches.

Results show the number of legend items, total confirmed matches, total possible
matches, per-item confidence, and a button to open each annotated PDF.

---

## API Reference

### `GET /health`

Returns server status.

```json
{
  "status": "ok",
  "api_key_set": true,
  "pdf2image": true,
  "pdfium": true,
  "tile_grid": "3x3"
}
```

### `POST /count`

**Form data:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `legend` | file | Yes | - | Legend/key PDF |
| `files[]` | file[] | Yes | - | One or more schematic PDFs |
| `dpi` | int | No | `200` | Render resolution |

**Response:**

```json
{
  "legend_items": [
    { "code": "EL-01", "description": "String Lighting" }
  ],
  "results_by_item": [
    {
      "code": "EL-01",
      "description": "String Lighting",
      "total": 14,
      "maybe": 2,
      "annotated_pdf": "base64-encoded-pdf",
      "pages": [
        {
          "page": 1,
          "count": 9,
          "maybe": 1,
          "confidence": "medium",
          "reasoning": "Tile-level reasoning summary.",
          "notes": ""
        }
      ]
    }
  ]
}
```

---

## Tips for Best Results

- Use 200+ DPI for dense or complex schematics.
- Use a clear legend/key page with visible symbol codes and descriptions.
- Remove unrelated legend pages if the model extracts too many non-countable
  entries.
- For very large drawings, crop or split to the relevant sheets before upload.
- Review the annotated PDFs rather than trusting totals blindly; coordinates and
  confidence are meant to make spot-checking easier.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `PORT` | `5050` | Server port |
| `DEBUG` | `0` | Flask debug mode (`0` or `1`) |

Do not commit real API keys. Use shell exports, EC2 environment configuration,
Docker Compose `.env` files that stay out of Git, or a secrets manager.

---

## Recommended Next Improvements

### Safety and deployment

- Replace the hard-coded frontend server URL with `http://localhost:5050` or
  `window.location.origin`.
- Restrict CORS in hosted deployments instead of allowing every origin.
- Add upload size, page count, legend item count, and DPI limits before model
  calls start.
- Return `400` for invalid `dpi` values and clamp accepted values to known
  options such as `100`, `150`, `200`, and `300`.
- For EC2 production use, run the Docker container behind Nginx/Caddy with HTTPS
  instead of exposing Flask directly to the internet.

### Accuracy

- Add tile overlap so labels near tile boundaries are less likely to be missed.
- Deduplicate nearby coordinates after tile merging to avoid double-counting
  boundary detections.
- Preserve schematic filenames in the backend response so multi-PDF results can
  be traced back to source files.

### User experience

- Add progress reporting. A full run can require:

  ```text
  legend items x schematic pages x 9 tile calls
  ```

  Without progress, long jobs look frozen.

- Show page-level details in the UI, not only the item totals.
- Offer a single combined report download in addition to one annotated PDF per
  item.

### Maintainability

- Pin dependency versions in `requirements.txt` once the app is stable.
- Move inline frontend JavaScript/CSS into separate files if the UI keeps
  growing.
- Build result rows with DOM APIs and `textContent` instead of `innerHTML`, since
  filenames and model-returned descriptions can contain unsafe text.
- Add a small test suite around PDF rendering, DPI validation, coordinate
  scaling, and result aggregation.
