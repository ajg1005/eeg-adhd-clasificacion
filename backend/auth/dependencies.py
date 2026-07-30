from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError

from backend.auth.security import decode_access_token
from backend.auth.service import get_active_user
from backend.db.models import User


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    try:
        user_id = decode_access_token(token)
    except InvalidTokenError as exc:
        raise _credentials_exception() from exc

    user = get_active_user(user_id)
    if user is None:
        raise _credentials_exception()
    return user


def _credentials_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se han podido validar las credenciales.",
        headers={"WWW-Authenticate": "Bearer"},
    )
