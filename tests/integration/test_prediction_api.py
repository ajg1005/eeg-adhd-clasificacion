from tests.conftest import requires_ml_model


# comprueba que /validate acepta un CSV de un paciente valido
@requires_ml_model
def test_validate_endpoint_accepts_sample_patient_csv(client, post_csv, sample_prediction_csv_path):
    response = post_csv(client, sample_prediction_csv_path, "/validate")

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["rows"] > 0
    assert data["available_channels"]


# comprueba que /validate rechaza un CSV al que le faltan canales obligatorios
@requires_ml_model
def test_validate_endpoint_rejects_missing_channels(client, post_csv, invalid_missing_columns_csv_path):
    response = post_csv(client, invalid_missing_columns_csv_path, "/validate")

    assert response.status_code == 400


# comprueba que /predict devuelve una clasificacion final (ADHD o Control)
@requires_ml_model
def test_predict_endpoint_returns_classification(client, post_csv, sample_prediction_csv_path):
    response = post_csv(client, sample_prediction_csv_path, "/predict")

    assert response.status_code == 200
    data = response.json()
    assert data["prediction_label"] in {"Control", "ADHD"}
    assert data["n_epochs"] > 0
