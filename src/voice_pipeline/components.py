from __future__ import annotations

import json
import subprocess
import tempfile
import urllib.request
import urllib.error
from urllib.parse import urlparse
from pathlib import Path


class WhisperTranscriber:
    def __init__(self, model: str = "small", device: str = "auto", compute_type: str = "int8") -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install the voice extra: pip install '.[voice]'") from exc
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> str:
        segments, _ = self.model.transcribe(str(audio_path), vad_filter=True, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://127.0.0.1:11434",
        system_prompt: str = "You are a concise, helpful voice assistant.",
        timeout: float = 120,
    ) -> None:
        parsed = urlparse(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment):
            raise ValueError("Ollama URL must be HTTP(S), without credentials, query, or fragment")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.system_prompt = system_prompt

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "options": {"temperature": 0.3},
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read(1_048_577)
                if len(raw) > 1_048_576:
                    raise RuntimeError("Ollama response exceeds 1 MB")
                payload = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("Could not reach Ollama; check the service and timeout") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("response"), str):
            raise RuntimeError("Ollama response must contain a text response field")
        answer = payload["response"].strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        return answer


class PiperSynthesizer:
    def __init__(self, model_path: Path, executable: str = "piper", timeout: float = 120) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        self.model_path = model_path
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout
        self.executable = executable

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Stage a fresh result; a failed run must not accept or overwrite an old WAV.
        with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
            staged = Path(directory) / "response.wav"
            command = [self.executable, "--model", str(self.model_path), "--output_file", str(staged)]
            try:
                subprocess.run(command, input=text.encode(), check=True,
                               capture_output=True, timeout=self.timeout)
            except FileNotFoundError as exc:
                raise RuntimeError("Piper executable was not found on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Piper synthesis timed out") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Piper synthesis failed; check the voice model and executable") from exc
            if not staged.is_file() or staged.stat().st_size == 0:
                raise RuntimeError("Piper did not produce a nonempty audio file")
            staged.replace(output_path)
        return output_path
