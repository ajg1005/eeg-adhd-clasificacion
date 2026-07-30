from datetime import timedelta
from uuid import uuid4

import pytest

from backend.auth import repository
from backend.auth.security import create_access_token, verify_password


PASSWORD = "password-segura"


def _unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid4().hex}@example.com"


def _register(client, email: str, password: str = PASSWORD):
    return client.post(
        "/auth/register",
        json={"email": email, "password": password},
    )


def test_register_user(client):
    email = _unique_email("register")

    response = _register(client, email.upper())

    assert response.status_code == 201
    assert response.json()["email"] == email
    assert response.json()["is_active"] is True
    assert "password" not in response.text

    stored_user = repository.get_user_by_email(email)
    assert stored_user is not None
    assert stored_user.password_hash != PASSWORD
    assert verify_password(PASSWORD, stored_user.password_hash)


def test_register_rejects_duplicate_email(client):
    email = _unique_email("duplicate")
    assert _register(client, email).status_code == 201

    response = _register(client, email.upper())

    assert response.status_code == 409
    assert response.json()["detail"] == "Ya existe un usuario con ese correo."


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("not-an-email", PASSWORD),
        ("short-password@example.com", "short"),
    ],
)
def test_register_validates_input(client, email, password):
    response = _register(client, email, password)

    assert response.status_code == 422


def test_login_and_get_current_user(client):
    email = _unique_email("login")
    register_response = _register(client, email)
    assert register_response.status_code == 201

    login_response = client.post(
        "/auth/login",
        data={"username": email.upper(), "password": PASSWORD},
    )

    assert login_response.status_code == 200
    assert login_response.json()["token_type"] == "bearer"

    token = login_response.json()["access_token"]
    profile_response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert profile_response.status_code == 200
    assert profile_response.json()["id"] == register_response.json()["id"]
    assert profile_response.json()["email"] == email


@pytest.mark.parametrize(
    ("email_factory", "password"),
    [
        (lambda registered_email: registered_email, "incorrecta"),
        (lambda _: _unique_email("unknown"), PASSWORD),
    ],
)
def test_login_rejects_invalid_credentials(client, email_factory, password):
    registered_email = _unique_email("credentials")
    assert _register(client, registered_email).status_code == 201

    response = client.post(
        "/auth/login",
        data={
            "username": email_factory(registered_email),
            "password": password,
        },
    )

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_current_user_rejects_missing_and_invalid_tokens(client):
    missing_response = client.get("/auth/me")
    invalid_response = client.get(
        "/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert missing_response.status_code == 401
    assert invalid_response.status_code == 401


def test_current_user_rejects_expired_token(client):
    email = _unique_email("expired")
    register_response = _register(client, email)
    user_id = register_response.json()["id"]
    token = create_access_token(user_id, expires_delta=timedelta(seconds=-1))

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 401
