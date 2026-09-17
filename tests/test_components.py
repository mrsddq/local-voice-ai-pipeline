import io
import json
import subprocess
from unittest.mock import Mock

import pytest

from voice_pipeline.components import OllamaClient, PiperSynthesizer
from voice_pipeline.pipeline import LocalVoicePipeline


@pytest.mark.parametrize("payload", [{"response": None}, {"response": 42}, [], {"response": " "}])
def test_ollama_rejects_malformed_payload(monkeypatch, payload):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(json.dumps(payload).encode()))
    with pytest.raises(RuntimeError):
        OllamaClient().generate("Hello")


def test_ollama_valid_text_and_bounded_read(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b'{"response":" Hello "}'))
    assert OllamaClient().generate("Hello") == "Hello"
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: io.BytesIO(b"a" * 1_048_577))
    with pytest.raises(RuntimeError, match="exceeds"):
        OllamaClient().generate("Hello")


def test_piper_timeout_is_explicit(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    run = Mock(side_effect=subprocess.TimeoutExpired("piper", 1))
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(RuntimeError, match="timed out"):
        PiperSynthesizer(model, timeout=1).synthesize("Hello", tmp_path / "out.wav")
    assert run.call_args.kwargs["timeout"] == 1


def test_piper_success_without_output_is_rejected(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setattr(subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="nonempty"):
        PiperSynthesizer(model).synthesize("Hello", tmp_path / "out.wav")


def test_pipeline_cannot_overwrite_input(tmp_path):
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"original")
    transcriber = Mock()
    with pytest.raises(ValueError, match="overwrite"):
        LocalVoicePipeline(transcriber, Mock(), Mock()).run(audio, audio)
    transcriber.transcribe.assert_not_called()
    assert audio.read_bytes() == b"original"


def test_piper_failure_preserves_existing_output(tmp_path, monkeypatch):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    output = tmp_path / "out.wav"
    output.write_bytes(b"old audio")
    monkeypatch.setattr(subprocess, "run", Mock())
    with pytest.raises(RuntimeError, match="nonempty"):
        PiperSynthesizer(model).synthesize("Hello", output)
    assert output.read_bytes() == b"old audio"


def test_piper_publishes_new_output_atomically(tmp_path, monkeypatch):
    from pathlib import Path
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    output = tmp_path / "out.wav"
    output.write_bytes(b"old")
    def run(command, **kwargs):
        Path(command[-1]).write_bytes(b"new audio")
    monkeypatch.setattr(subprocess, "run", run)
    assert PiperSynthesizer(model).synthesize("Hello", output) == output
    assert output.read_bytes() == b"new audio"
