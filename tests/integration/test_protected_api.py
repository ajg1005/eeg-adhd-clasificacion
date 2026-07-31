import pytest


@pytest.mark.parametrize(
    "path",
    [
        "/models",
        "/training/options",
        "/training/datasets",
        "/experiments",
        "/tasks/task-id",
    ],
)
def test_private_get_endpoints_require_authentication(client, path):
    response = client.get(path)

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_inference_requires_authentication(client):
    response = client.post(
        "/validate",
        files={"file": ("sample.csv", b"ID,Class\n", "text/csv")},
    )

    assert response.status_code == 401


def test_training_requires_authentication(client):
    response = client.post(
        "/training/run",
        data={
            "model_type": "ml",
            "model_name": "random_forest",
            "dataset_id": "1",
        },
    )

    assert response.status_code == 401
