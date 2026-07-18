"""Engine selection from settings."""

from inference_server.config import Settings
from inference_server.engines.base import BaseInferenceEngine
from inference_server.engines.echo import EchoEngine
from inference_server.engines.onnx import OnnxEngine


def get_inference_engine(settings: Settings) -> BaseInferenceEngine:
    """Build the engine backend named by `settings.engine`.

    Raises ValueError for unknown engine names so misconfiguration
    fails at startup rather than on the first request.
    """
    if settings.engine == "echo":
        return EchoEngine()
    if settings.engine == "onnx":
        return OnnxEngine(settings)
    raise ValueError(f"Unknown inference engine: {settings.engine!r}")
