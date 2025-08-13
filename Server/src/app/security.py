# src/app/security.py

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from src.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Checks if the provided API key is valid."""
    if api_key_header == settings.api_key:
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )