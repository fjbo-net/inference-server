from pathlib import Path

from inference_server.engines.onnx import discover_models


def _make_model_folder(
    models_dir: Path,
    name: str,
    with_config: bool = True
) -> None:
    folder = models_dir / name
    folder.mkdir(parents=True)
    if with_config:
        (folder / "genai_config.json").write_text("{}")


def test_discover_models_returns_model_folders_when_config_is_present(
    tmp_path: Path
) -> None:
    # Arrange
    expected_models = [
        "phi-3.5-mini",
        "qwen2.5-0.5b"
    ]

    _make_model_folder(
        tmp_path,
        "qwen2.5-0.5b"
    )
    _make_model_folder(
        tmp_path,
        "phi-3.5-mini"
    )


    # Act
    models = discover_models(tmp_path)


    # Assert
    assert models == expected_models


def test_discover_models_ignores_folders_when_config_is_missing(
    tmp_path: Path
) -> None:
    # Arrange
    expected_models = ["real-model"]

    _make_model_folder(
        tmp_path,
        "real-model"
    )
    _make_model_folder(
        tmp_path,
        "just-some-folder",
        with_config=False
    )
    (tmp_path / "loose-file.onnx").write_text("")


    # Act
    models = discover_models(tmp_path)


    # Assert
    assert models == expected_models


def test_discover_models_returns_empty_list_when_models_dir_is_missing(
    tmp_path: Path
) -> None:
    # Arrange
    missing_dir = tmp_path / "does-not-exist"


    # Act
    models = discover_models(missing_dir)


    # Assert
    assert models == []
