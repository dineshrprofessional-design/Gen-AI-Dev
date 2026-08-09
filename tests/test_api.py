from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_item_crud():
    created = client.post("/api/v1/items", json={"name": "Widget", "price": 9.99})
    assert created.status_code == 201
    item_id = created.json()["id"]

    assert client.get(f"/api/v1/items/{item_id}").json()["name"] == "Widget"

    patched = client.patch(f"/api/v1/items/{item_id}", json={"price": 12.5})
    assert patched.json()["price"] == 12.5

    assert client.delete(f"/api/v1/items/{item_id}").status_code == 204
    assert client.get(f"/api/v1/items/{item_id}").status_code == 404


def test_validation_rejects_bad_price():
    assert client.post("/api/v1/items", json={"name": "X", "price": 0}).status_code == 422
