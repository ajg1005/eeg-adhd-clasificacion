from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.db.engine import SessionLocal
from backend.db.models import User


class DuplicateUserEmailError(Exception):
    """Indica que el correo ya pertenece a otro usuario."""


def get_user_by_email(email: str) -> User | None:
    with SessionLocal() as session:
        query = select(User).where(User.email == email)
        return session.scalar(query)


def get_user_by_id(user_id: int) -> User | None:
    with SessionLocal() as session:
        return session.get(User, user_id)


def create_user(email: str, password_hash: str) -> User:
    with SessionLocal() as session:
        user = User(email=email, password_hash=password_hash)
        session.add(user)

        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise DuplicateUserEmailError(email) from exc

        session.refresh(user)
        return user
