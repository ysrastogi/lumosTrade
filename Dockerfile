FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose API and WebSocket ports
EXPOSE 8000
EXPOSE 8765

# Set environment variables
ENV PYTHONPATH=/app

# Run API server
CMD ["python", "run_api.py"]
