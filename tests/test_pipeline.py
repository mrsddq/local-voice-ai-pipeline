from pathlib import Path

import pytest

from voice_pipeline import LocalVoicePipeline


class FakeTranscriber:
    def transcribe(self, audio_path):
        return "What time does the library close?"


class FakeModel:
    def generate(self, prompt):
        assert "library" in prompt
        return "The library closes at six."


class FakeSynthesizer:
    def synthesize(self, text, output_path):
        assert text.endswith("six.")
        return output_path


def test_pipeline_connects_all_three_stages(tmp_path):
    audio = tmp_path / "question.wav"
    audio.write_bytes(b"RIFF-fake")
    output = tmp_path / "response.wav"
    pipeline = LocalVoicePipeline(FakeTranscriber(), FakeModel(), FakeSynthesizer())
    result = pipeline.run(audio, output)
    assert result.transcript.startswith("What time")
    assert result.response.endswith("six.")
    assert Path(result.audio_path) == output


def test_missing_audio_fails_before_components_run(tmp_path):
    pipeline = LocalVoicePipeline(FakeTranscriber(), FakeModel(), FakeSynthesizer())
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "missing.wav", tmp_path / "out.wav")

