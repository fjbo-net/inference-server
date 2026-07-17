import pytest
from fastapi.testclient import TestClient

from inference_server.app import create_app
from inference_server.config import Settings


@pytest.fixture
def client() -> TestClient:
    settings = Settings(
        _env_file=None,
        engine="echo"
    )
    return TestClient(create_app(settings))


def test_list_models_returns_engine_models(client: TestClient) -> None:
    # Arrange
    expected_status_code = 200
    expected_model_ids = ["echo"]


    # Act
    response = client.get("/v1/models")


    # Assert
    body = response.json()
    assert response.status_code == expected_status_code
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == expected_model_ids
