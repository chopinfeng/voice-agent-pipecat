"""Headless test of the full STT -> LLM -> TTS chain, no microphone needed.

Stage 1 feeds a WAV through local Whisper and prints the transcription.
Stage 2 sends that transcription to the OpenRouter LLM and speaks the reply
through local Piper, writing the result to a WAV.

Run with:
    uv run --project pipecat python openrouter_smoke_test.py
"""

import asyncio
import os
import wave
from pathlib import Path

from dotenv import load_dotenv

from pipecat.frames.frames import (
    InputAudioRawFrame,
    LLMContextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX
from pipecat.tests.utils import SleepFrame, run_test

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

IN_WAV = HERE / "smoke_test_output.wav"
OUT_WAV = HERE / "openrouter_reply.wav"
MODEL_DIR = HERE / "piper-voices"
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")


class TextCollector(FrameProcessor):
    """Records LLM text as it passes between the LLM and the TTS service."""

    def __init__(self):
        super().__init__()
        self.chunks: list[str] = []

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMTextFrame):
            self.chunks.append(frame.text)
        await self.push_frame(frame, direction)

    @property
    def text(self) -> str:
        return "".join(self.chunks).strip()


def read_wav(path: Path) -> tuple[bytes, int, int]:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes()), wav.getframerate(), wav.getnchannels()


def write_wav(path: Path, pcm: bytes, sample_rate: int, channels: int) -> float:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return len(pcm) / (sample_rate * channels * 2)


async def transcribe() -> str:
    pcm, sample_rate, channels = read_wav(IN_WAV)

    # Chunk the audio the way a transport would deliver it (20 ms frames).
    chunk = int(sample_rate * 0.02) * 2 * channels
    audio_frames = [
        InputAudioRawFrame(
            audio=pcm[i : i + chunk], sample_rate=sample_rate, num_channels=channels
        )
        for i in range(0, len(pcm), chunk)
    ]

    stt = WhisperSTTServiceMLX(
        settings=WhisperSTTServiceMLX.Settings(model=MLXModel.LARGE_V3_TURBO_Q4.value)
    )
    down_frames, _ = await run_test(
        stt,
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            *audio_frames,
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=20.0),
        ],
    )

    text = " ".join(f.text for f in down_frames if isinstance(f, TranscriptionFrame)).strip()
    assert text, "Whisper produced no transcription"
    return text


async def respond(user_text: str) -> tuple[str, bytes, int, int]:
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=LLM_MODEL,
            system_instruction=(
                "You are a voice assistant. Reply in one short spoken sentence."
            ),
        ),
    )
    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice="en_US-ryan-high"),
    )

    context = LLMContext()
    context.add_message({"role": "user", "content": user_text})

    collector = TextCollector()
    down_frames, _ = await run_test(
        Pipeline([llm, collector, tts]),
        frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=25.0)],
    )

    reply = collector.text
    audio = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert reply, "LLM produced no text"
    assert audio, "TTS produced no audio"
    pcm = b"".join(f.audio for f in audio)
    return reply, pcm, audio[0].sample_rate, audio[0].num_channels


async def main():
    transcription = await transcribe()
    reply, pcm, sample_rate, channels = await respond(transcription)
    duration = write_wav(OUT_WAV, pcm, sample_rate, channels)

    print("\n" + "=" * 70)
    print(f"STT (local Whisper) : {transcription}")
    print(f"LLM ({LLM_MODEL}) : {reply}")
    print(f"TTS (local Piper)   : {duration:.2f}s -> {OUT_WAV}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
