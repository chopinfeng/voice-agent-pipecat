"""填充语：用户说完之后先应一声，把等待藏起来。

管线里剩下的耗时几乎全在 LLM 的「网络 + 模型」那一段，约 1.5 秒，换模型也压不下去。
但用户感知的延迟不是 LLM 什么时候出第一个 token，而是**自己说完之后多久听到声音**。
所以在等待期间先播一句极短的应答词，感知延迟从 2.8 秒降到 1.05 秒，真正的回答随后
接上——音频是顺序播放的，两段不会重叠。

放在 **TTS 之后**、输出 transport 之前：

    Pipeline([..., llm, tts, FillerSpeech(...), transport.output(), assistant_aggregator])

这个位置有两个关键好处：

* 填充语以预先合成好的音频帧注入，不带任何文本帧，所以不会被助手聚合器当成模型的
  回答写进上下文（放在 TTS 之前推 ``TTSSpeakFrame`` 就会——上下文里会多出一句
  ``assistant: 嗯，``，下一轮直接把对话带偏）。
* 真实回答开始合成时 TTS 会推 ``TTSStartedFrame``，正好当作撤销信号；填充语自己
  不经过 TTS，不会误伤自己。

音频在管线启动时一次合成好，触发时只是把字节推下去，不占用合成时间。
"""

import asyncio
import itertools
from collections.abc import Sequence
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_PHRASES = ("嗯，", "好的，", "我看看，")


def synthesize_clips(
    *,
    download_dir: Path,
    voice: str,
    phrases: Sequence[str],
    engine: str = "piper",
) -> tuple[list[bytes], int]:
    """预先把填充语合成成 PCM。

    直接调合成库而不是 pipecat 的 TTS 服务——这些音频不该经过管线，只是一段随时
    可以推出去的字节。

    **引擎要跟正文的 TTS 一致**，否则一轮里会出现两个嗓子：先用 A 的声音应一声「嗯」，
    紧接着 B 的声音开始答，听起来像换了个人。这里合成是在启动时做一次并缓存的，
    所以用慢一点的引擎也不影响响应速度。

    Args:
        download_dir: Piper 语音模型所在目录，``engine="kokoro"`` 时用不到。
        voice: 语音名。Piper 是 ``zh_CN-huayan-medium`` 这种，Kokoro 是 ``zf_xiaoxiao``。
        phrases: 要合成的短语。
        engine: ``piper`` 或 ``kokoro``。

    Returns:
        (每条短语的 16 位 PCM, 采样率)。
    """
    if engine == "kokoro":
        return _kokoro_clips(voice, phrases)

    from piper import PiperVoice
    from piper.download_voices import download_voice

    path = download_dir / f"{voice}.onnx"
    if not path.exists():
        download_voice(voice, download_dir)
    loaded = PiperVoice.load(str(path))

    clips = [
        b"".join(c.audio_int16_bytes for c in loaded.synthesize(p)) for p in phrases
    ]
    return clips, loaded.config.sample_rate


def _kokoro_clips(voice: str, phrases: Sequence[str]) -> tuple[list[bytes], int]:
    """用 Kokoro 合成填充语。

    复用 pipecat 那个服务的下载逻辑（模型和音色包放在 ``~/.cache/pipecat``），
    但直接调底层的 ``Kokoro``，不走管线。
    """
    import numpy as np
    from kokoro_onnx import Kokoro
    from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR, _ensure_model_files

    model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    _ensure_model_files(model, voices)
    kokoro = Kokoro(str(model), str(voices))

    clips, rate = [], 0
    for phrase in phrases:
        samples, rate = kokoro.create(phrase, voice=voice, speed=1.0, lang="cmn")
        clips.append((np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return clips, rate


class FillerSpeech(FrameProcessor):
    """轮次结束后，若 LLM 迟迟不出词就先播一段预合成的应答语。"""

    def __init__(
        self,
        *,
        clips: Sequence[bytes],
        sample_rate: int,
        delay: float = 0.25,
        **kwargs,
    ):
        """初始化。

        Args:
            clips: 预合成的 16 位单声道 PCM，轮流使用，避免听起来像卡带。
            sample_rate: ``clips`` 的采样率。
            delay: 等这么久还没等到真实回答才播。设大一点可以让快回答完全不触发
                填充语，设小一点则更早出声。
            **kwargs: 透传给 FrameProcessor。
        """
        super().__init__(**kwargs)
        self._clips = itertools.cycle(clips)
        self._sample_rate = sample_rate
        self._delay = delay
        self._pending = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """轮次结束时上膛，真实回答一开始合成（或被打断）就撤销。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStoppedSpeakingFrame):
            self._arm()
        elif isinstance(frame, (TTSStartedFrame, InterruptionFrame, StartFrame)):
            await self._disarm()

        await self.push_frame(frame, direction)

    def _arm(self):
        if self._pending is None:
            self._pending = self.create_task(self._speak_later())

    async def _disarm(self):
        task, self._pending = self._pending, None
        if task:
            await self.cancel_task(task)

    async def _speak_later(self):
        await asyncio.sleep(self._delay)
        self._pending = None
        logger.debug(f"{self}: 真实回答还没到，先播一段填充语")
        await self.push_frame(
            TTSAudioRawFrame(
                audio=next(self._clips), sample_rate=self._sample_rate, num_channels=1
            )
        )
