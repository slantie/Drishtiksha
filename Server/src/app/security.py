# src/app/security.py

import secrets
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from src.config import settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """
    Checks if the provided API key is valid using a constant-time comparison
    to prevent timing attacks.
    """
    # Retrieve the secret value from the Pydantic SecretStr
    correct_api_key = settings.api_key.get_secret_value()

    if api_key_header and secrets.compare_digest(api_key_header, correct_api_key):
        return api_key_header
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )