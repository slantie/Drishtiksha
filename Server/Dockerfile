# --- STAGE 1: Build the Python environment with uv ---
FROM python:3.12-slim AS builder

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv, the fast package manager
RUN pip install uv

WORKDIR /app

COPY pyproject.toml .

# Create a virtual environment and install dependencies into it
RUN uv venv && uv sync

# --- STAGE 2: Create the final production image ---
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# FIX: Create a non-root user and group for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv ./.venv

# Add the virtual environment's bin directory to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy your application source code
# FIX: Change ownership of the app code to the new user
COPY --chown=appuser:appgroup . .

# FIX: Switch to the non-root user
USER appuser

EXPOSE 8000

# The command to start the Uvicorn server
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]