# src/app/security.py

import secrets
import logging
from fastapi import Security, HTTPException, status, Request
from fastapi.security import APIKeyHeader

from src.config import settings

# REFACTOR: Get a logger instance for this module.
logger = logging.getLogger(__name__)

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(
    request: Request, # REFACTOR: Add the Request object as a dependency to access client info.
    api_key_header: str = Security(API_KEY_HEADER)
):
    """
    Checks if the provided API key is valid using a constant-time comparison
    to prevent timing attacks. It now also logs failed authentication attempts.
    """
    correct_api_key = settings.api_key.get_secret_value()
    client_ip = request.client.host if request.client else "unknown"

    if api_key_header and secrets.compare_digest(api_key_header, correct_api_key):
        # The key is valid, return it to proceed with the request.
        return api_key_header
    else:
        # --- REFACTOR: Log the failed attempt before raising an exception ---
        log_message = f"Unauthorized API access attempt from IP: {client_ip}. "
        if not api_key_header:
            log_message += "Reason: Missing API Key."
        else:
            # IMPORTANT: Do NOT log the provided key itself to prevent credential leakage.
            log_message += "Reason: Invalid API Key provided."
        
        logger.warning(log_message)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )