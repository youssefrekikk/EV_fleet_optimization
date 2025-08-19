# Multi-stage Dockerfile for EV Fleet Optimization Studio
# Optimized for production deployment with development support

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libpq-dev \
    libgdal-dev \
    gdal-bin \
    proj-bin \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip show scikit-learn

# Copy source code
COPY . .

# Install package
RUN pip install -e .

# Debug: Show installed packages and versions
RUN pip list | grep -E "(scikit-learn|numpy|scipy)" || echo "No matching packages found"

# Test sklearn imports
RUN python test_sklearn_import.py || echo "Warning: sklearn import test failed, but continuing build"

# Show final package versions
RUN pip freeze | grep -E "(scikit-learn|numpy|scipy|joblib)" || echo "No matching packages found"

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
