# Local Voice AI Pipeline

A modular, private voice assistant turn that connects local speech recognition, a local language model, and local speech synthesis:

`audio file -> faster-whisper -> Ollama -> Piper -> response.wav`

Audio and prompts stay on the machine after the required models are downloaded.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) with a model such as `llama3.2:3b`
- [Piper](https://github.com/rhasspy/piper) executable and an `.onnx` voice model
- FFmpeg available on `PATH`

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[voice,dev]"
ollama pull llama3.2:3b

local-voice question.wav \
  --piper-model models/en_US-lessac-medium.onnx \
  --output response.wav
```

The command prints the transcript, model response, generated file, and elapsed time as JSON. Use `--whisper-model`, `--ollama-model`, and `--ollama-url` to tune the pipeline.

## Design

Each stage implements a tiny protocol and can be swapped independently. Tests use fakes, so CI does not download multi-gigabyte models. The Ollama request is non-streaming for a simple one-turn baseline; a real-time UI can consume streaming tokens and synthesize sentence-by-sentence.

## Docker

The image includes FFmpeg and faster-whisper. Mount a Piper binary/model and point the container at the host Ollama service. GPU and audio-device passthrough are host-specific, so the native setup is usually faster for local development.

```bash
pytest
ruff check .
docker build -t local-voice-ai .
```

MIT licensed.

## Adapter reliability and measured scope

Ollama calls use a configurable timeout, a 1 MB response budget, JSON/type validation, and
nonempty text checks. Piper calls have a configurable timeout and require a nonempty output file.
The pipeline refuses to overwrite the input audio. Offline tests fake the transport and subprocess
boundaries to exercise malformed responses, download budgets, timeouts, missing output, and stage
ordering. Passing these tests demonstrates orchestration behavior, not speech recognition quality,
audio quality, model accuracy, real-time performance, or GPU throughput.

The default Ollama endpoint is loopback. A custom `--ollama-url` can send prompts to another host,
so the local-privacy statement applies only when that service runs locally. Live operation still
requires separately installed Whisper weights, Ollama, and Piper voices. CI does not download or
execute those models. Authentication and concurrency controls are not provided by this CLI.
