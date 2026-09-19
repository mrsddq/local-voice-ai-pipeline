from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


class LanguageModel(Protocol):
    def generate(self, prompt: str) -> str: ...


class Synthesizer(Protocol):
    def synthesize(self, text: str, output_path: Path) -> Path: ...


@dataclass(frozen=True)
class PipelineResult:
    transcript: str
    response: str
    audio_path: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class LocalVoicePipeline:
    def __init__(self, transcriber: Transcriber, model: LanguageModel, synthesizer: Synthesizer) -> None:
        self.transcriber = transcriber
        self.model = model
        self.synthesizer = synthesizer

    def run(self, input_path: Path, output_path: Path) -> PipelineResult:
        if not input_path.is_file():
            raise FileNotFoundError(f"Audio input not found: {input_path}")
        if input_path.resolve() == output_path.resolve():
            raise ValueError("Output path must not overwrite the input audio")
        started = time.perf_counter()
        transcript = self.transcriber.transcribe(input_path).strip()
        if not transcript:
            raise ValueError("No speech was detected in the audio")
        response = self.model.generate(transcript).strip()
        if not response:
            raise ValueError("The language model returned an empty response")
        rendered = self.synthesizer.synthesize(response, output_path)
        return PipelineResult(
            transcript=transcript,
            response=response,
            audio_path=str(rendered),
            elapsed_seconds=round(time.perf_counter() - started, 3),
        )

