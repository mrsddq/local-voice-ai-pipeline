from __future__ import annotations

import json
import subprocess
import urllib.request
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
    ) -> None:
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
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc
        answer = str(payload.get("response", "")).strip()
        if not answer:
            raise RuntimeError("Ollama returned an empty response")
        return answer


class PiperSynthesizer:
    def __init__(self, model_path: Path, executable: str = "piper") -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")
        self.model_path = model_path
        self.executable = executable

    def synthesize(self, text: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.executable,
            "--model", str(self.model_path),
            "--output_file", str(output_path),
        ]
        try:
            subprocess.run(command, input=text.encode(), check=True, capture_output=True)
        except FileNotFoundError as exc:
            raise RuntimeError("Piper executable was not found on PATH") from exc
        except subprocess.CalledProcessError as exc:
            error = exc.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"Piper failed: {error}") from exc
        return output_path

