"""Offline smoke test for the local Pipecat install.

Runs a real Pipecat pipeline with the fully local Piper TTS service (no cloud
API keys) and writes the synthesized audio to a WAV file. Verifies that frame
plumbing, the service base classes and audio resampling all work end to end.

Run with:
    uv run --project pipecat python local_smoke_test.py
"""

import asyncio
import wave
from pathlib import Path

from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame
from pipecat.services.piper.tts import PiperTTSService
from pipecat.tests.utils import run_test

TEXT = "Hello from Pipecat. This audio was synthesized locally, with no cloud API key."
OUT_WAV = Path(__file__).parent / "smoke_test_output.wav"
MODEL_DIR = Path(__file__).parent / "piper-voices"


async def main():
    MODEL_DIR.mkdir(exist_ok=True)

    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice="en_US-ryan-high"),
    )

    down_frames, _ = await run_test(
        tts,
        frames_to_send=[TTSSpeakFrame(TEXT)],
    )

    audio = [f for f in down_frames if isinstance(f, TTSAudioRawFrame)]
    assert audio, "no TTSAudioRawFrame produced"

    pcm = b"".join(f.audio for f in audio)
    sample_rate = audio[0].sample_rate
    channels = audio[0].num_channels

    with wave.open(str(OUT_WAV), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)

    duration = len(pcm) / (sample_rate * channels * 2)
    print(f"\nOK: {len(audio)} audio frames, {len(pcm)} bytes, {sample_rate} Hz")
    print(f"OK: wrote {duration:.2f}s of audio to {OUT_WAV}")


if __name__ == "__main__":
    asyncio.run(main())
