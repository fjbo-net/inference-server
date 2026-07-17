from fastapi.testclient import TestClient

from inference_server import __version__
from inference_server.app import create_app


def test_health() -> None:
    # Arrange
    app = create_app()
    client = TestClient(app)
    

    # Act
    response = client.get("/health")


    # Assert
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
