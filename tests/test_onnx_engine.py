import asyncio
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_server.config import Settings
from inference_server.engines.base import (
    EngineError,
    GenerationChunk,
    ModelNotFoundError,
)
from inference_server.engines.onnx import OnnxEngine
from tests.factories import (
    make_chat_completion_request,
    make_user_message_payload,
)

FAKE_MODEL_ID = "fake-model"
FAKE_PROMPT_TOKEN_COUNT = 4


def _collect_chunks(
    stream: AsyncIterator[GenerationChunk]
) -> list[GenerationChunk]:
    async def _consume() -> list[GenerationChunk]:
        return [chunk async for chunk in stream]

    return asyncio.run(_consume())


def _make_fake_runtime(words: list[str]) -> SimpleNamespace:
    created_configs: list[Any] = []

    class FakeConfig:
        def __init__(self, path: str) -> None:
            self.path = path
            self.providers: list[str] = []
            created_configs.append(self)

        def clear_providers(self) -> None:
            self.providers = []

        def append_provider(self, provider: str) -> None:
            self.providers.append(provider)

    class FakeModel:
        def __init__(self, source: Any) -> None:
            self.source = source

    class FakeTokenizerStream:
        def decode(self, token: int) -> str:
            word = words[token]
            return word if token == 0 else f" {word}"

    class FakeTokenizer:
        def __init__(self, model: FakeModel) -> None:
            self.model = model

        def apply_chat_template(
            self,
            messages: str,
            add_generation_prompt: bool
        ) -> str:
            return messages

        def encode(self, prompt: str) -> list[int]:
            return [0] * FAKE_PROMPT_TOKEN_COUNT

        def create_stream(self) -> FakeTokenizerStream:
            return FakeTokenizerStream()

    class FakeGeneratorParams:
        def __init__(self, model: FakeModel) -> None:
            self.search_options: dict[str, Any] = {}

        def set_search_options(self, **options: Any) -> None:
            self.search_options = options

    class FakeGenerator:
        def __init__(
            self,
            model: FakeModel,
            params: FakeGeneratorParams
        ) -> None:
            self._next = 0
            self._current = 0

        def append_tokens(self, tokens: list[int]) -> None:
            self.input_tokens = tokens

        def is_done(self) -> bool:
            return self._next >= len(words)

        def generate_next_token(self) -> None:
            self._current = self._next
            self._next += 1

        def get_next_tokens(self) -> list[int]:
            return [self._current]

    return SimpleNamespace(
        Config=FakeConfig,
        Model=FakeModel,
        Tokenizer=FakeTokenizer,
        GeneratorParams=FakeGeneratorParams,
        Generator=FakeGenerator,
        created_configs=created_configs
    )


def _make_engine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    words: list[str],
    device: str = "cpu"
) -> OnnxEngine:
    model_dir = tmp_path / FAKE_MODEL_ID
    model_dir.mkdir(parents=True)
    (model_dir / "genai_config.json").write_text("{}")
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime_genai",
        _make_fake_runtime(words)
    )
    settings = Settings(
        _env_file=None,
        engine="onnx",
        device=device,
        models_dir=tmp_path
    )
    return OnnxEngine(settings)


def test_onnx_engine_lists_discovered_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    expected_models = [FAKE_MODEL_ID]

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=["hi"]
    )


    # Act
    models = engine.list_models()


    # Assert
    assert models == expected_models


def test_onnx_engine_streams_tokens_when_model_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    expected_text = "hello from onnx"
    expected_finish_reason = "stop"

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=expected_text.split()
    )
    request = make_chat_completion_request(model=FAKE_MODEL_ID)


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    tokens = [chunk.token for chunk in chunks if chunk.token is not None]
    assert "".join(tokens) == expected_text
    assert chunks[-1].finish_reason == expected_finish_reason


def test_onnx_engine_truncates_when_max_tokens_is_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    expected_completion_tokens = 2
    expected_finish_reason = "length"

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=["one", "two", "three", "four"]
    )
    request = make_chat_completion_request(
        model=FAKE_MODEL_ID,
        max_tokens=expected_completion_tokens
    )


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    tokens = [chunk.token for chunk in chunks if chunk.token is not None]
    assert len(tokens) == expected_completion_tokens
    assert chunks[-1].finish_reason == expected_finish_reason


def test_onnx_engine_reports_usage_when_generation_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    words = ["counting", "real", "tokens"]
    expected_prompt_tokens = FAKE_PROMPT_TOKEN_COUNT
    expected_completion_tokens = len(words)

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=words
    )
    request = make_chat_completion_request(model=FAKE_MODEL_ID)


    # Act
    chunks = _collect_chunks(engine.generate(request))


    # Assert
    usage = chunks[-1].usage
    assert usage is not None
    assert usage.prompt_tokens == expected_prompt_tokens
    assert usage.completion_tokens == expected_completion_tokens
    assert usage.total_tokens == expected_prompt_tokens + expected_completion_tokens


def test_onnx_engine_appends_qnn_provider_when_device_is_qnn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    expected_providers = ["qnn"]

    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=["hi"],
        device="qnn"
    )
    request = make_chat_completion_request(model=FAKE_MODEL_ID)


    # Act
    _collect_chunks(engine.generate(request))


    # Assert
    fake_runtime = sys.modules["onnxruntime_genai"]
    assert fake_runtime.created_configs[0].providers == expected_providers


def test_onnx_engine_raises_value_error_when_device_is_unknown(
    tmp_path: Path
) -> None:
    # Arrange
    settings = Settings(
        _env_file=None,
        engine="onnx",
        device="warp-core",
        models_dir=tmp_path
    )


    # Act & Assert
    with pytest.raises(ValueError):
        OnnxEngine(settings)


def test_onnx_engine_raises_model_not_found_when_model_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=["hi"]
    )
    request = make_chat_completion_request(model="not-downloaded")


    # Act & Assert
    with pytest.raises(ModelNotFoundError):
        engine.generate(request)


def test_onnx_engine_raises_engine_error_when_runtime_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    engine = _make_engine(
        tmp_path,
        monkeypatch,
        words=["hi"]
    )
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime_genai",
        None
    )
    request = make_chat_completion_request(
        model=FAKE_MODEL_ID,
        messages=[make_user_message_payload(content="anyone there?")]
    )


    # Act & Assert
    with pytest.raises(EngineError):
        _collect_chunks(engine.generate(request))
