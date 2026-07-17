import sys
from pathlib import Path

import pytest

from inference_server.config import Settings, get_base_dir


def test_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)

    expected_host = "127.0.0.1"
    expected_port = 8000
    expected_models_dir = tmp_path / "models"


    # Act
    settings = Settings(_env_file=None)


    # Assert
    assert settings.host == expected_host
    assert settings.port == expected_port
    assert settings.models_dir == expected_models_dir


def test_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    monkeypatch.setenv("INFERENCE_HOST", "0.0.0.0")
    monkeypatch.setenv("INFERENCE_PORT", "9001")
    monkeypatch.setenv("INFERENCE_MODELS_DIR", str(tmp_path / "external-models"))


    # Act
    settings = Settings(_env_file=None)


    # Assert
    assert settings.host == "0.0.0.0"
    assert settings.port == 9001
    assert settings.models_dir == tmp_path / "external-models"


def test_base_dir_unfrozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert get_base_dir() == tmp_path


def test_base_dir_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_exe = tmp_path / "dist" / "inference-server.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    assert get_base_dir() == fake_exe.parent
