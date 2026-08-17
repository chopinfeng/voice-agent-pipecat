"""Measure per-stage latency for the voice pipeline.

Times each leg separately — STT, LLM time-to-first-token, TTS time-to-first-audio —
so it is clear which one dominates. Every candidate LLM is measured twice: the
first call pays TLS handshake cost, the second reuses the connection.

Run with:
    uv run --project pipecat python bench.py
"""

import asyncio
import os
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

IN_WAV = HERE / "zh_test_input.wav"
MODEL_DIR = HERE / "piper-voices"
PROMPT = "用一句话介绍杭州。"


class Mark(FrameProcessor):
    """Records the wall-clock time a frame type first passes through."""

    def __init__(self, watch: type[Frame]):
        super().__init__()
        self._watch = watch
        self.at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.at is None and isinstance(frame, self._watch):
            self.at = time.perf_counter()
        await self.push_frame(frame, direction)


def audio_frames() -> list[InputAudioRawFrame]:
    with wave.open(str(IN_WAV), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        rate, channels = wav.getframerate(), wav.getnchannels()
    chunk = int(rate * 0.02) * 2 * channels
    return [
        InputAudioRawFrame(
            audio=pcm[i : i + chunk], sample_rate=rate, num_channels=channels
        )
        for i in range(0, len(pcm), chunk)
    ]


async def _stt_pass(stt, frames: list[InputAudioRawFrame]) -> tuple[float, str]:
    start = Mark(VADUserStoppedSpeakingFrame)
    done = Mark(TranscriptionFrame)
    out, _ = await run_test(
        Pipeline([start, stt, done]),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            *frames,
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=30.0),
        ],
    )
    text = " ".join(f.text for f in out if isinstance(f, TranscriptionFrame)).strip()
    elapsed = (done.at - start.at) if (done.at and start.at) else float("nan")
    return elapsed, text


async def bench_stt(name: str, stt, frames: list[InputAudioRawFrame]):
    # First pass pays model download / lazy load; only the second pass is timed.
    await _stt_pass(stt, frames)
    elapsed, text = await _stt_pass(stt, frames)
    print(f"  STT  {name:38s} {elapsed:6.2f}s  {text}")
    return elapsed


async def bench_llm(model: str, runs: int = 2):
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=model,
            system_instruction="你是语音助手，用一到两句简短的中文口语回答。",
        ),
    )
    timings = []
    for i in range(runs):
        context = LLMContext()
        context.add_message({"role": "user", "content": PROMPT})
        start = Mark(LLMContextFrame)
        done = Mark(LLMTextFrame)
        await run_test(
            Pipeline([start, llm, done]),
            frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=20.0)],
        )
        elapsed = (done.at - start.at) if (done.at and start.at) else float("nan")
        timings.append(elapsed)
        label = "cold" if i == 0 else f"warm{i}"
        print(f"  LLM  {model:30s} {label:6s} {elapsed:6.2f}s")
    return timings


async def bench_tts(voice: str):
    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice=voice),
    )
    start = Mark(TTSSpeakFrame)
    done = Mark(TTSAudioRawFrame)
    # Warm the model, then time a second synthesis.
    await run_test(Pipeline([tts]), frames_to_send=[TTSSpeakFrame("预热。")])
    await run_test(
        Pipeline([start, tts, done]),
        frames_to_send=[TTSSpeakFrame("杭州是浙江省的省会。"), SleepFrame(sleep=10.0)],
    )
    elapsed = (done.at - start.at) if (done.at and start.at) else float("nan")
    print(f"  TTS  {voice:38s} {elapsed:6.2f}s")
    return elapsed


async def main():
    frames = audio_frames()
    print("\n--- STT (中文音频 3.5s) ---")

    from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX

    await bench_stt(
        "MLX whisper large-v3-turbo-q4",
        WhisperSTTServiceMLX(
            settings=WhisperSTTServiceMLX.Settings(
                model=MLXModel.LARGE_V3_TURBO_Q4.value, language=Language.ZH
            )
        ),
        frames,
    )
    await bench_stt(
        "MLX whisper tiny",
        WhisperSTTServiceMLX(
            settings=WhisperSTTServiceMLX.Settings(
                model=MLXModel.TINY.value, language=Language.ZH
            )
        ),
        frames,
    )

    from pipecat.services.funasr.stt import FunASRSTTService

    await bench_stt(
        "FunASR SenseVoiceSmall",
        FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH)),
        frames,
    )

    print("\n--- LLM (OpenRouter, TTFT) ---")
    for model in [
        "openai/gpt-4.1-mini",
        "openai/gpt-4.1-nano",
        "google/gemini-2.5-flash-lite",
    ]:
        await bench_llm(model)

    print("\n--- TTS (首个音频帧) ---")
    await bench_tts("zh_CN-huayan-medium")


if __name__ == "__main__":
    asyncio.run(main())
