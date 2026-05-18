from tests.conftest import requires_ml_model


# comprueba que /models lista los modelos habilitados (ml_best y dl_best)
def test_models_endpoint_lists_enabled_models(client):
    response = client.get("/models")

    assert response.status_code == 200
    model_ids = {model["model_id"] for model in response.json()["models"]}
    assert {"ml_best", "dl_best"}.issubset(model_ids)


# comprueba que /model/catalog agrupa los modelos por familia (ml/dl)
def test_model_catalog_endpoint_groups_training_models(client):
    response = client.get("/model/catalog")

    assert response.status_code == 200
    data = response.json()
    assert data["machine_learning"]
    assert data["deep_learning"]


# comprueba que /model/info devuelve metadatos del modelo seleccionado
@requires_ml_model
def test_model_info_endpoint_returns_metadata(client):
    response = client.get("/model/info", params={"model_id": "ml_best"})

    assert response.status_code == 200
    data = response.json()
    assert data["model_id"] == "ml_best"
    assert data["channels"]


# comprueba que /model/info devuelve 404 para un modelo que no existe
def test_model_info_endpoint_rejects_unknown_model(client):
    response = client.get("/model/info", params={"model_id": "unknown"})

    assert response.status_code == 404
