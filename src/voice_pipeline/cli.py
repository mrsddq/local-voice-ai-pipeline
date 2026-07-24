import argparse
import json
from pathlib import Path

from .components import OllamaClient, PiperSynthesizer, WhisperTranscriber
from .pipeline import LocalVoicePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fully local voice assistant turn")
    parser.add_argument("input", type=Path, help="Input audio file supported by faster-whisper")
    parser.add_argument("--output", "-o", type=Path, default=Path("response.wav"))
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--piper-model", type=Path, required=True, help="Path to a Piper .onnx model")
    parser.add_argument("--piper-executable", default="piper")
    args = parser.parse_args()

    pipeline = LocalVoicePipeline(
        WhisperTranscriber(args.whisper_model),
        OllamaClient(args.ollama_model, args.ollama_url),
        PiperSynthesizer(args.piper_model, args.piper_executable),
    )
    result = pipeline.run(args.input, args.output)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

