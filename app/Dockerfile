# LegalLLM Docker Container
# Multi-stage build for smaller image size

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy application code
COPY app/ ./app/
COPY data/dataset/train.json ./data/dataset/train.json
COPY results/eval_results.json ./results/eval_results.json

# Expose ports
EXPOSE 8000 8501

# Start both FastAPI and Streamlit
COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]