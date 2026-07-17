from fastapi.testclient import TestClient

from inference_server import __version__
from inference_server.app import create_app


def test_health_returns_ok_and_version() -> None:
    # Arrange
    app = create_app()
    client = TestClient(app)

    expected_status_code = 200
    expected_response = {
        "status": "ok",
        "version": __version__
    }


    # Act
    response = client.get("/health")


    # Assert
    assert response.status_code == expected_status_code
    assert response.json() == expected_response
