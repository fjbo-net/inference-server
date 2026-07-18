"""ONNX Runtime GenAI engine backend (Qualcomm NPU via QNN, CPU for dev)."""

import asyncio
import importlib
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from inference_server.config import Settings
from inference_server.engines.base import (
    BaseInferenceEngine,
    EngineError,
    GenerationChunk,
    ModelNotFoundError,
)
from inference_server.schemas.openai import (
    ChatCompletionRequest,
    FinishReason,
    Usage,
)

GENAI_CONFIG_FILENAME = "genai_config.json"
DEFAULT_MAX_TOKENS = 1024
SUPPORTED_DEVICES = (
    "cpu",
    "qnn"
)


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


class OnnxEngine(BaseInferenceEngine):
    def __init__(self, settings: Settings) -> None:
        if settings.device not in SUPPORTED_DEVICES:
            raise ValueError(
                f"Unknown device for the onnx engine: {settings.device!r} "
                f"(supported: {', '.join(SUPPORTED_DEVICES)})"
            )
        self._models_dir = settings.models_dir
        self._device = settings.device
        self._runtime: Any | None = None
        self._loaded: dict[str, tuple[Any, Any]] = {}

    @property
    def name(self) -> str:
        return "onnx"

    def list_models(self) -> list[str]:
        return discover_models(self._models_dir)

    def generate(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        # Validate eagerly so errors surface before streaming starts.
        if request.model not in self.list_models():
            raise ModelNotFoundError(request.model)
        return self._stream(request)

    def _runtime_module(self) -> Any:
        if self._runtime is None:
            try:
                self._runtime = importlib.import_module("onnxruntime_genai")
            except ImportError as error:
                raise EngineError(
                    "onnxruntime-genai is not installed; run `uv sync --group onnx`."
                ) from error
        return self._runtime

    def _load(self, model_id: str) -> tuple[Any, Any]:
        if model_id not in self._loaded:
            runtime = self._runtime_module()
            model_path = str(self._models_dir / model_id)
            if self._device == "cpu":
                model = runtime.Model(model_path)
            else:
                config = runtime.Config(model_path)
                config.clear_providers()
                config.append_provider(self._device)
                model = runtime.Model(config)
            tokenizer = runtime.Tokenizer(model)
            self._loaded[model_id] = (
                model,
                tokenizer
            )
        return self._loaded[model_id]

    async def _stream(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        try:
            async for chunk in self._generate_chunks(request):
                yield chunk
        except Exception as error:
            # A failure mid-stream must surface as a chunk: the HTTP
            # response has already started, so raising would sever it.
            yield GenerationChunk(error=str(error))

    async def _generate_chunks(
        self,
        request: ChatCompletionRequest
    ) -> AsyncIterator[GenerationChunk]:
        runtime = self._runtime_module()
        model, tokenizer = await asyncio.to_thread(
            self._load,
            request.model
        )
        prompt = self._build_prompt(
            tokenizer,
            request
        )
        input_tokens = tokenizer.encode(prompt)
        max_tokens = (
            request.max_tokens
            if request.max_tokens is not None
            else DEFAULT_MAX_TOKENS
        )

        params = runtime.GeneratorParams(model)
        params.set_search_options(
            **self._search_options(
                request,
                len(input_tokens) + max_tokens
            )
        )
        generator = runtime.Generator(
            model,
            params
        )
        generator.append_tokens(input_tokens)
        token_stream = tokenizer.create_stream()

        completion_tokens = 0
        while not generator.is_done() and completion_tokens < max_tokens:
            await asyncio.to_thread(generator.generate_next_token)
            completion_tokens += 1
            for token in generator.get_next_tokens():
                text = token_stream.decode(token)
                if text:
                    yield GenerationChunk(token=text)

        finish_reason: FinishReason = (
            "stop"
            if generator.is_done()
            else "length"
        )
        prompt_tokens = len(input_tokens)
        yield GenerationChunk(
            finish_reason=finish_reason,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )

    def _build_prompt(
        self,
        tokenizer: Any,
        request: ChatCompletionRequest
    ) -> str:
        messages = json.dumps(
            [
                {
                    "role": message.role,
                    "content": message.content or ""
                }
                for message in request.messages
            ]
        )
        try:
            prompt = tokenizer.apply_chat_template(
                messages=messages,
                add_generation_prompt=True
            )
        except AttributeError:
            # Older onnxruntime-genai builds lack apply_chat_template.
            lines = [
                f"{message.role}: {message.content or ''}"
                for message in request.messages
            ]
            prompt = "\n".join(lines) + "\nassistant:"
        return str(prompt)

    def _search_options(
        self,
        request: ChatCompletionRequest,
        max_length: int
    ) -> dict[str, Any]:
        options: dict[str, Any] = {"max_length": max_length}
        if request.temperature is not None:
            options["temperature"] = request.temperature
            options["do_sample"] = True
        if request.top_p is not None:
            options["top_p"] = request.top_p
            options["do_sample"] = True
        return options
