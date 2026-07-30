from datetime import datetime, timedelta, timezone

import jwt
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash

from backend.core.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)


MIN_SECRET_LENGTH = 32
password_hash = PasswordHash.recommended()


def hash_password(password: str) -> str:
    """Genera un hash Argon2 para una contrasena."""
    return password_hash.hash(password)


def verify_password(password: str, stored_hash: str) -> bool:
    """Comprueba una contrasena contra el hash almacenado."""
    return password_hash.verify(password, stored_hash)


def create_access_token(
    user_id: int,
    expires_delta: timedelta | None = None,
) -> str:
    """Crea un token de acceso firmado para un usuario."""
    if user_id < 1:
        raise ValueError("El identificador de usuario debe ser positivo.")

    now = datetime.now(timezone.utc)
    expires_at = now + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {
        "sub": str(user_id),
        "iat": now,
        "exp": expires_at,
    }
    return jwt.encode(payload, _secret_key(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> int:
    """Valida un token y devuelve el identificador del usuario."""
    payload = jwt.decode(
        token,
        _secret_key(),
        algorithms=[JWT_ALGORITHM],
        options={"require": ["sub", "iat", "exp"]},
    )

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidTokenError("El token no contiene un usuario valido.") from exc
    if user_id < 1:
        raise InvalidTokenError("El token no contiene un usuario valido.")
    return user_id


def _secret_key() -> str:
    if JWT_SECRET_KEY is None or len(JWT_SECRET_KEY) < MIN_SECRET_LENGTH:
        raise RuntimeError("JWT_SECRET_KEY debe contener al menos 32 caracteres.")
    return JWT_SECRET_KEY
