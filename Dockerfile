# TPUv6-ZeroNAS Production Docker Image
FROM python:3.9-slim

LABEL maintainer="daniel@terragonlabs.com"
LABEL description="TPUv6-ZeroNAS: Neural Architecture Search for TPUv6 Optimization"
LABEL version="0.1.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY tpuv6_zeronas/ ./tpuv6_zeronas/
COPY examples/ ./examples/
COPY scripts/ ./scripts/
COPY README.md .
COPY LICENSE .

# Install the package
RUN pip install -e .

# Create directories for cache and logs
RUN mkdir -p /app/.cache /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TPUV6_CACHE_DIR=/app/.cache
ENV TPUV6_LOG_DIR=/app/logs

# Create non-root user for security
RUN useradd -m -u 1000 tpuv6user && chown -R tpuv6user:tpuv6user /app
USER tpuv6user

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import tpuv6_zeronas; print('OK')" || exit 1

# Default command
CMD ["python", "examples/basic_search.py"]

# Expose port for future API integration
EXPOSE 8080