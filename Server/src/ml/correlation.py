# src/ml/correlation.py

import uuid
import logging
from contextvars import ContextVar
from typing import Optional

# Context variable to store correlation ID per request
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


def generate_correlation_id() -> str:
    """
    Generate a new correlation ID.
    
    Returns:
        str: A new UUID-based correlation ID
    """
    return str(uuid.uuid4())


def get_correlation_id() -> str:
    """
    Get the current correlation ID. If none exists, generate a new one.
    
    Returns:
        str: The correlation ID for the current context
    """
    correlation_id = correlation_id_var.get()
    if correlation_id is None:
        correlation_id = generate_correlation_id()
        correlation_id_var.set(correlation_id)
    return correlation_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set a specific correlation ID for the current context.
    
    Args:
        correlation_id: The correlation ID to set
    """
    correlation_id_var.set(correlation_id)


def reset_correlation_id() -> None:
    """Reset the correlation ID (useful for testing or cleanup)."""
    correlation_id_var.set(None)


class CorrelationLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes correlation ID in all log messages.
    
    Usage:
        logger = CorrelationLoggerAdapter(logging.getLogger(__name__))
        logger.info("Processing started")  # Automatically includes correlation_id
    """
    
    def process(self, msg, kwargs):
        """Add correlation ID to the log message."""
        correlation_id = get_correlation_id()
        return f"[CID: {correlation_id}] {msg}", kwargs


def get_logger(name: str) -> CorrelationLoggerAdapter:
    """
    Get a logger with automatic correlation ID tracking.
    
    Args:
        name: The logger name (typically __name__)
        
    Returns:
        CorrelationLoggerAdapter: A logger that includes correlation IDs
    """
    base_logger = logging.getLogger(name)
    return CorrelationLoggerAdapter(base_logger, {})
