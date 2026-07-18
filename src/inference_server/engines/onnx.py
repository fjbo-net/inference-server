"""ONNX Runtime GenAI engine backend (Qualcomm NPU via QNN, CPU for dev)."""

from pathlib import Path

GENAI_CONFIG_FILENAME = "genai_config.json"


def discover_models(models_dir: Path) -> list[str]:
    """Return names of model folders that contain a `genai_config.json`.

    ONNX Runtime GenAI models are directories; the config file marks a
    folder as a loadable model. A missing models dir yields an empty
    list so a fresh install starts cleanly.
    """
    if not models_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in models_dir.iterdir()
        if entry.is_dir() and (entry / GENAI_CONFIG_FILENAME).is_file()
    )
