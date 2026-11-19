# Container for the PCE prediction GUI (e.g. Hugging Face Spaces "Docker" SDK, Render, Railway).
# Build from the repository root:   docker build -f deploy/Dockerfile -t pce-gui .
# Run locally:                      docker run -p 7860:7860 pce-gui   ->  http://localhost:7860
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt deploy/requirements-gui.txt ./
RUN pip install --no-cache-dir -r requirements-gui.txt
COPY app.py ./
COPY templates ./templates
COPY static ./static
COPY models ./models
COPY models_v2 ./models_v2
COPY revision/rev ./revision/rev
ENV PORT=7860
EXPOSE 7860
CMD gunicorn -w 2 -b 0.0.0.0:${PORT} app:app
