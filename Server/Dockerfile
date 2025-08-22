# --- STAGE 1: Build the Python environment with uv ---
# This image is based on Debian, so we use 'apt-get' not 'apk'.
FROM python:3.12-slim AS builder

# CORRECTED: Install system dependencies using Debian's package manager 'apt-get'
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install uv, the fast package manager
RUN pip install uv

# Set the working directory
WORKDIR /app

# Copy the file that defines your project dependencies
COPY pyproject.toml .

# Create a virtual environment and install dependencies into it
RUN uv venv && uv sync

# --- STAGE 2: Create the final production image ---
FROM python:3.12-slim

# Set environment variables for best practices in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the virtual environment with all installed packages from the builder stage
COPY --from=builder /app/.venv ./.venv

# Add the virtual environment's bin directory to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy your application source code
COPY . .

# Expose the port that the FastAPI server will run on
EXPOSE 8000

# The command to start the Uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]