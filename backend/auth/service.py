from backend.auth import repository
from backend.auth.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from backend.db.models import User


class EmailAlreadyRegisteredError(Exception):
    """Indica que el correo ya esta registrado."""


_DUMMY_PASSWORD_HASH = hash_password("dummy-password-for-timing-check")


def register_user(email: str, password: str) -> User:
    normalized_email = _normalize_email(email)
    if repository.get_user_by_email(normalized_email) is not None:
        raise EmailAlreadyRegisteredError(normalized_email)

    try:
        return repository.create_user(
            email=normalized_email,
            password_hash=hash_password(password),
        )
    except repository.DuplicateUserEmailError as exc:
        raise EmailAlreadyRegisteredError(normalized_email) from exc


def authenticate_user(email: str, password: str) -> User | None:
    user = repository.get_user_by_email(_normalize_email(email))
    if user is None:
        verify_password(password, _DUMMY_PASSWORD_HASH)
        return None

    password_is_valid = verify_password(password, user.password_hash)
    if not password_is_valid or not user.is_active:
        return None
    return user


def get_active_user(user_id: int) -> User | None:
    user = repository.get_user_by_id(user_id)
    if user is None or not user.is_active:
        return None
    return user


def create_user_access_token(user: User) -> str:
    return create_access_token(int(user.id))


def _normalize_email(email: str) -> str:
    return email.strip().casefold()
