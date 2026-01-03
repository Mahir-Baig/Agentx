# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Create a temporary requirements file without the -e . line
RUN grep -v '^-e \.' requirements.txt > temp_requirements.txt || true

# Install Python dependencies (excluding -e .)
RUN pip install --no-cache-dir -r temp_requirements.txt

# Copy the entire application
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Create data directories
RUN mkdir -p data/chromadb data/logs

# Expose ports
# 8000 for FastAPI
# 8501 for Streamlit (if needed)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Run FastAPI
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
