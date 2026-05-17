FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY data/ ./data/
COPY scraper.py .

# Pre-build vector index at image build time (optional — speeds up cold start)
# RUN python -c "from app.retriever import retriever; retriever.load()"

# Create vectorstore directory
RUN mkdir -p vectorstore

# Expose port
EXPOSE 8000

# Environment variables (override at runtime)
ENV LLM_PROVIDER=gemini
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
