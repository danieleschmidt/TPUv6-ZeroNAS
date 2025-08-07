# TPUv6-ZeroNAS Docker Image
# Multi-stage build for optimized production deployment

# Base stage with Python
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and source code
COPY requirements.txt requirements-minimal.txt ./
COPY tpuv6_zeronas/ ./tpuv6_zeronas/
COPY setup.py README.md LICENSE ./
COPY scripts/ ./scripts/

# Development stage
FROM base as development

# Install full dependencies for development
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -e ".[full,dev]"

# Run validation tests
RUN python scripts/simple_integration_test.py

CMD ["python", "-m", "tpuv6_zeronas.cli", "--help"]

# Production stage (minimal dependencies)
FROM base as production

# Install minimal dependencies only
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN groupadd -r tpunas && useradd -r -g tpunas tpunas
RUN chown -R tpunas:tpunas /app
USER tpunas

# Validate installation
RUN python scripts/quick_test_minimal.py

# Default command
CMD ["python", "-m", "tpuv6_zeronas.cli", "search", "--help"]

# Minimal stage (smallest possible)
FROM python:3.11-alpine as minimal

WORKDIR /app

# Copy only essential files
COPY tpuv6_zeronas/ ./tpuv6_zeronas/
COPY setup.py ./

# Install without dependencies
RUN pip install --no-deps -e .

# Create user
RUN addgroup -S tpunas && adduser -S tpunas -G tpunas
RUN chown -R tpunas:tpunas /app
USER tpunas

CMD ["python", "-c", "from tpuv6_zeronas import ArchitectureSpace; print('TPUv6-ZeroNAS ready')"]
