# Dockerfile is the recipe for building one portable image of this app.
# Think of it as "a tiny Linux machine with exactly the tools this project needs."

# Start from an official Python image. The "slim" variant is smaller than the
# full image but still has Debian package management available.
FROM python:3.11-slim

# Keep Python logs unbuffered so container logs show up immediately.
ENV PYTHONUNBUFFERED=1

# Inside the container, /app will be the project folder.
WORKDIR /app

# pdf2image needs Poppler binaries to turn PDF pages into images.
# --no-install-recommends keeps the image smaller by avoiding optional packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first. Docker caches layers, so future builds can skip
# reinstalling packages when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the actual application files into the image.
COPY server.py index.html ./

# Flask reads PORT in server.py. This documents the intended internal port.
EXPOSE 5050

# This is the command that runs when a container starts from this image.
CMD ["python", "server.py"]
