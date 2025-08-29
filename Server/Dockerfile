# Dockerfile

# --- STAGE 1: The Builder ---
FROM python:3.12.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .

RUN pip install uv && \
    uv venv && \
    uv sync

# --- STAGE 2: The Final Production Image ---
FROM python:3.12.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    # FIX: Set environment variables to point caching libraries to a writable directory.
    # This prevents them from trying to write to the non-existent root home directory.
    HF_HOME="/app/cache/huggingface" \
    MPLCONFIGDIR="/app/cache/matplotlib" \
    XDG_CACHE_HOME="/app/cache"

# Install required RUNTIME system libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group for enhanced security.
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy the virtual environment from the builder stage.
COPY --from=builder /app/.venv ./.venv

# Copy the application source code.
COPY --chown=appuser:appgroup . .

# FIX: Create the cache directory and give our non-root user ownership.
RUN mkdir -p /app/cache/huggingface /app/cache/matplotlib && \
    chown -R appuser:appgroup /app/cache

# Switch to the non-root user.
USER appuser

EXPOSE 8000

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]