# comprueba que el endpoint /health responde 200 con status ok
def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
