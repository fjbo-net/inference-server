# Verifying the ONNX engine

Two passes: CPU on any dev machine (validates the engine against the real
`onnxruntime-genai` API), then QNN on a Snapdragon X device (validates the
NPU path). Both end with the same SDK check.

## 1. CPU pass (any machine)

```sh
uv sync --group onnx
```

Download an ONNX Runtime GenAI model into `models/` — the folder must contain
`genai_config.json`. Known-good CPU option (about 2 GB):

```sh
# from https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
# copy the cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4 folder to:
models/phi-3-mini-4k-instruct/
```

Run and check:

```sh
INFERENCE_ENGINE=onnx uv run python -m inference_server
curl http://127.0.0.1:8000/v1/models
```

## 2. QNN pass (Snapdragon X laptop)

Prerequisites: Windows on ARM64, a **native ARM64** Python 3.12, and a model
in ONNX Runtime GenAI format built for the Hexagon NPU — note that Qualcomm
AI Hub LLMs target Genie, not this engine; use
[onnx-community/Llama-3.2-3B-instruct-hexagon-npu-assets](https://huggingface.co/onnx-community/Llama-3.2-3B-instruct-hexagon-npu-assets)
(or the `llmware` equivalent) placed under `models/`.

**Check the interpreter architecture first** — uv may install an x86_64
Python on ARM64 machines, which runs emulated and can never load `win_arm64`
wheels or reach the NPU:

```sh
uv run python -c "import platform, sys; print(platform.machine(), sys.version)"
```

The build tag must say `(ARM64)`, not `(AMD64)` — `platform.machine()` alone
reports the OS architecture and can say `ARM64` even under an emulated
interpreter. If it shows `(AMD64)`:

```sh
uv python install cpython-3.12-windows-aarch64-none
rm -rf .venv
uv venv --python cpython-3.12-windows-aarch64-none
```

```sh
uv sync --group onnx    # on ARM64 this also installs the onnxruntime-qnn plugin
INFERENCE_ENGINE=onnx INFERENCE_DEVICE=qnn uv run python -m inference_server
```

`onnxruntime-qnn` >= 2.0 is a plugin execution provider: it installs
*alongside* the standard `onnxruntime` package (which stays), and the QNN
provider does not appear in `onnxruntime.get_available_providers()` until it
is registered at runtime — the server does that automatically when
`INFERENCE_DEVICE=qnn`. To sanity-check the plugin install by hand:

```sh
uv run python -c "import onnxruntime as ort, onnxruntime_qnn; ort.register_execution_provider_library('QNNExecutionProvider', onnxruntime_qnn.get_library_path()); print('plugin OK')"
```

## 3. SDK check (both passes)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="unused"
)

model_id = client.models.list().data[0].id

stream = client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": "Say hello in five words."}],
    max_tokens=32,
    stream=True
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(
            delta.content,
            end="",
            flush=True
        )
print()
```

Expected: tokens stream incrementally; the final chunk carries
`finish_reason` (`"length"` if the 32-token cap was hit). On the QNN pass,
Task Manager should show NPU utilization during generation.

## What to check when something fails

- `503`/startup `ValueError` — unknown `INFERENCE_ENGINE`/`INFERENCE_DEVICE` value.
- `{"error": {..., "code": "model_not_found"}}` — the model folder has no
  `genai_config.json` or `INFERENCE_MODELS_DIR` points elsewhere.
- `{"error": {..., "code": "engine_error"}}` mentioning `onnxruntime-genai is
  not installed` — the `onnx` dependency group is missing.
- `engine_error` mentioning `QNNExecutionProvider is unavailable` — the
  `onnxruntime-qnn` plugin is not installed on this device.
- `engine_error` with `JSON Error: model:decoder: Unknown value
  "sliding_window_key_value_cache"` — the model's `genai_config.json` uses a
  pre-mainline schema name; rename that key to `sliding_window` (same
  sub-keys) and retry. Cascading unknown-key errors mean the assets need a
  newer/nightly `onnxruntime-genai`, or use a differently packaged model
  (e.g. `llmware/llama-3.2-3b-onnx-qnn`).
- NPU-format models (QNN context binaries) only run on Snapdragon hardware —
  on x86 dev machines use a CPU-format model (section 1).
- SSE stream ends with `data: {"error": ...}` before `[DONE]` — the runtime
  failed mid-generation; the message carries the underlying cause.
