import sys
from pathlib import Path

import pytest

from inference_server.config import Settings, get_base_dir


def test_settings_uses_defaults_when_env_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
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


def test_env_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    # Arrange
    expected_host = "0.0.0.0"
    expected_port = 9001
    expected_models_dir = tmp_path / "external-models"

    monkeypatch.setenv(
        "INFERENCE_HOST",
        expected_host
    )
    monkeypatch.setenv(
        "INFERENCE_PORT",
        str(expected_port)
    )
    monkeypatch.setenv(
        "INFERENCE_MODELS_DIR",
        str(expected_models_dir)
    )


    # Act
    settings = Settings(_env_file=None)


    # Assert
    assert settings.host == expected_host
    assert settings.port == expected_port
    assert settings.models_dir == expected_models_dir


def test_base_dir_unfrozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    # Arrange
    monkeypatch.chdir(tmp_path)


    # Act
    base_dir = get_base_dir()


    # Assert
    assert base_dir == tmp_path


def test_base_dir_frozen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path
) -> None:
    # Arrange
    fake_exe = tmp_path / "dist" / "inference-server.exe"
    monkeypatch.setattr(
        sys,
        "frozen",
        True,
        raising=False
    )
    monkeypatch.setattr(
        sys,
        "executable",
        str(fake_exe)
    )


    # Act
    base_dir = get_base_dir()


    # Assert
    assert base_dir == fake_exe.parent
