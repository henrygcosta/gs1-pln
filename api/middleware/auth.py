"""
api/middleware/auth.py
Autenticação via API Key no header X-API-Key.
"""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from config.settings import get_settings
from domain.exceptions import AuthenticationError


async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> None:
    """Dependency que valida a API Key no header da requisição."""
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
