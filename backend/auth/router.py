from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from backend.auth.dependencies import get_current_user
from backend.auth.schemas import TokenResponse, UserCreate, UserResponse
from backend.auth.service import (
    EmailAlreadyRegisteredError,
    authenticate_user,
    create_user_access_token,
    register_user,
)
from backend.db.models import User


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(payload: UserCreate) -> User:
    try:
        return register_user(str(payload.email), payload.password)
    except EmailAlreadyRegisteredError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ya existe un usuario con ese correo.",
        ) from exc


@router.post("/login", response_model=TokenResponse)
def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenResponse:
    user = authenticate_user(form_data.username, form_data.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Correo o contrasena incorrectos.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return TokenResponse(access_token=create_user_access_token(user))


@router.get("/me", response_model=UserResponse)
def get_profile(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    return current_user
