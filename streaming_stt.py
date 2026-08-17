"""边说边转的本地听写。**默认不用**——实测比分段更慢，原因见下。

想法是这样的：默认的 ``FunASRSTTService`` 是 ``SegmentedSTTService``，必须等 VAD 判定
说完才把整段交给模型，所以「说完到拿到文本」= VAD 判定 0.67 秒 + 转写 0.74 秒。流式
边到边转，等 VAD 宣布说完时文本应该已经在手上，那 0.74 秒就从关键路径上消失了。

**离线探针支持这个想法，接进全链路就不成立了。**同一段 3.5 秒音频，
``realtime_replay.py`` 真 VAD 真节奏（听写 / 轮次判定 / 说完到出声，中位）：

===============  ========  ========  ========
方案              听写       轮次判定    说完→出声
===============  ========  ========  ========
分段 SenseVoice   0.70 秒   0.71 秒    0.96 秒
流式 960 毫秒片    2.06 秒   2.47 秒    2.98 秒
流式 1440 毫秒片   0.89 秒   1.29 秒    1.90 秒
流式 1920 毫秒片   0.61 秒   1.01 秒    1.71 秒
===============  ========  ========  ========

两个原因叠在一起：单片推理约 535 毫秒且**几乎与片长无关**（固定开销），而离线探针里
那 44% 的实时余量在真实链路里被 VAD、TTS、听写抢 CPU 吃掉了，积压下来到判定说完时
还有好几片没转完。加大分片能减少推理次数、把积压压下去，但片越大就越退化成分段，
而一句话通常只有两三秒，根本没有足够的说话时间摊薄这笔固定开销——流式要划算，
得是长句子加上更快的推理。

留着是因为长句场景（口述、朗读）可能反过来，用 ``STT=streaming`` 打开。日常对话别用。

模型是 FunASR 的 ``paraformer-zh-streaming``，分片大小是唯一要紧的参数。**分片不能
太小**：300 毫秒那档不光跟不上实时，识别本身就垮了（「请拥有一句话接绍一下自己的个
城城市」），上下文太短。600 毫秒起才准。默认 960 毫秒，实测最快的是 1920 毫秒，
但那时它已经基本等同于分段了。

推理是同步且吃 CPU 的，必须扔到线程里跑，否则每片会把事件循环堵住半秒，语音链路上
所有东西都跟着卡。
"""

import asyncio
import os
from collections.abc import AsyncGenerator

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

MODEL_RATE = 16000
# 分片长度 = chunk_size[1] × 60 毫秒。
CHUNK_UNITS = int(os.getenv("STREAM_CHUNK_UNITS", "16"))
CHUNK_SECS = CHUNK_UNITS * 0.06


class FunASRStreamingSTTService(STTService):
    """FunASR 流式听写。

    每收满一个分片就转一次，中间结果以 ``InterimTranscriptionFrame`` 发出；VAD 判定
    说完时把尾巴冲掉，再发一条完整的 ``TranscriptionFrame``。
    """

    def __init__(
        self,
        *,
        model: str = "paraformer-zh-streaming",
        language: Language = Language.ZH,
        chunk_units: int = CHUNK_UNITS,
        **kwargs,
    ):
        """初始化。

        Args:
            model: FunASR 模型名。
            language: 识别语言，只用于给出去的帧打标。
            chunk_units: 分片单位数，每个单位 60 毫秒。小于 10 会明显掉准确率。
            **kwargs: 透传给 ``STTService``。
        """
        super().__init__(**kwargs)
        self._model_name = model
        self._language = language
        self._units = chunk_units
        self._stride = chunk_units * 960  # 16k 下一个分片的采样点数
        self._model = None
        self._cache: dict = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""

    async def start(self, frame):
        """加载模型。第一次要下载约 880MB。"""
        await super().start(frame)
        if self._model is None:
            from funasr import AutoModel

            self._model = await asyncio.to_thread(
                AutoModel, model=self._model_name, disable_update=True
            )
            logger.debug(f"{self}: 流式模型就绪，分片 {self._units * 60} 毫秒")

    def can_generate_metrics(self) -> bool:
        """能报处理耗时。"""
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """开口时把上一句的状态清掉，别让缓存串到下一句。"""
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._reset()
        await super().process_frame(frame, direction)

    def _reset(self) -> None:
        self._cache = {}
        self._buf = np.zeros(0, dtype=np.float32)
        self._text = ""

    def _resample(self, pcm: bytes) -> np.ndarray:
        """转成 16k 单声道 float32。"""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rate = self.sample_rate or MODEL_RATE
        if rate == MODEL_RATE or len(x) == 0:
            return x
        idx = np.linspace(0, len(x) - 1, int(len(x) * MODEL_RATE / rate))
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    async def _infer(self, seg: np.ndarray, is_final: bool) -> str:
        """跑一片推理。同步且吃 CPU，扔线程里，不能堵事件循环。"""
        half = max(self._units // 2, 1)
        result = await asyncio.to_thread(
            lambda: self._model.generate(
                input=seg,
                cache=self._cache,
                is_final=is_final,
                chunk_size=[0, self._units, half],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )
        )
        return result[0]["text"] if result and result[0].get("text") else ""

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """攒够一个分片就转一次，吐中间结果。"""
        if self._model is None:
            yield None
            return

        self._buf = np.concatenate([self._buf, self._resample(audio)])
        while len(self._buf) >= self._stride:
            seg, self._buf = self._buf[: self._stride], self._buf[self._stride :]
            try:
                piece = await self._infer(seg, is_final=False)
            except Exception as e:  # noqa: BLE001 - 一片失败不该让整句作废
                logger.warning(f"{self}: 分片转写失败 {type(e).__name__}: {e}")
                continue
            if piece:
                self._text += piece
                yield InterimTranscriptionFrame(
                    self._text, self._user_id, time_now_iso8601(), self._language
                )

    async def _handle_vad_user_stopped_speaking(self, frame):
        """VAD 说停了：把尾巴冲掉，发完整结果。

        这一步是关键路径。理想情况下走到这里只剩不足一片的尾音，但实测积压往往还有
        好几片没转完，整条链路要等它们排完——这正是流式在短句上跑输分段的地方。
        """
        await super()._handle_vad_user_stopped_speaking(frame)
        if self._model is None:
            return
        try:
            tail = await self._infer(self._buf, is_final=True)
        except Exception as e:  # noqa: BLE001 - 收尾失败要报给链路，不能静默丢句
            await self.push_frame(ErrorFrame(f"流式听写收尾失败：{e}"))
            self._reset()
            return

        text = (self._text + tail).strip()
        self._reset()
        if text:
            logger.debug(f"Transcription: [{text}]")
            await self.push_frame(
                TranscriptionFrame(
                    text, self._user_id, time_now_iso8601(), self._language
                )
            )
