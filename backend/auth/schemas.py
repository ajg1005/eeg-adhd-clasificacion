from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from backend.api.schemas import OrmSchema


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserResponse(OrmSchema):
    id: int
    email: EmailStr
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
