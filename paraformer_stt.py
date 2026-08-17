"""带热词的中文听写：FunASR Paraformer contextual。

SenseVoice 快、中文底子好，但**没有热词机制**——项目里的专有名词它没法优先考虑，
「填充语」听成「春雨」，下游 agent 就去查一个乱码词。

Paraformer 的 contextual 变体支持热词：把项目里的文件名和术语作为提示传进去，
识别时会优先往这些词上靠。同一批台词实测 SenseVoice 对 3/6、这里对 4/6，
修好的是「填充语」（原听成「填充与」）和「进度接口」（原听成「精度接口」）。

**热词只对中文词有效。**英文标识符照样听不出来——「所有 python 文件」两个模型都听成
「所有掰了文件」，把 python 加进热词表也没用，只是从「掰了」变成「拜了」。原因是中文
ASR 的输出词表里根本没有英文，热词只能在中文候选之间做偏置，变不出一个英文串来。
要治英文得换中英混合模型，不是调热词能解决的。

代价：模型比 SenseVoice 大（首次下载约三分钟），单次推理慢 0.15 秒左右。

    stt = ParaformerSTTService(hotwords=["填充语", "调度器", "观察者"])
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

try:
    from funasr import AutoModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('需要 funasr：uv sync --extra funasr')
    raise ImportError(f"Missing module: {e}") from e

DEFAULT_MODEL = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"


@dataclass
class ParaformerSTTSettings(STTSettings):
    """ParaformerSTTService 的设置。"""

    pass


# 常用中文技术词。英文词放进来没有意义（模型输出不了英文），留着 python、pipecat
# 只是标记一下这两个词在语音里根本指望不上。
COMMON_TERMS = [
    "填充语", "过滤器", "调度器", "观察者", "进度接口", "延迟测试",
    "语音识别", "语音合成", "并发", "热词", "转写", "听写", "打断", "挂起",
]


def project_hotwords(root: Path, extra: list[str] | None = None) -> list[str]:
    """从项目目录里凑一份热词表。

    文件名去掉扩展名就是最容易被听岔的那批词（filler、scheduler、progress），
    加上一批常用技术词和调用方补充的术语。

    Args:
        root: 项目根目录。
        extra: 额外的术语。

    Returns:
        去重后的热词列表。
    """
    names = {p.stem for p in root.glob("*.py") if p.is_file()}
    return sorted(names | set(COMMON_TERMS) | set(extra or []))


class ParaformerSTTService(SegmentedSTTService):
    """带热词的本地中文听写。

    热词以空格分隔的字符串传给 FunASR，每次识别都带上——它是识别时的偏置，
    不是后处理替换，所以不会像词表那样误伤正常词。
    """

    Settings = ParaformerSTTSettings

    @property
    def wants_wav_segments(self) -> bool:
        """要原始 16 位 PCM，模型自己吃。"""
        return False

    def __init__(
        self,
        *,
        hotwords: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        settings: Settings | None = None,
        **kwargs,
    ):
        """初始化。

        Args:
            hotwords: 热词，识别时优先往这些词上靠。
            model: FunASR 模型名，默认是支持热词的 contextual 变体。
            settings: 运行时可改的设置。
            **kwargs: 透传给 SegmentedSTTService。
        """
        default = self.Settings(model=model, language=None)
        if settings is not None:
            default.apply_update(settings)
        super().__init__(settings=default, **kwargs)

        self._hotword = " ".join(hotwords or []) or None
        logger.debug(f"加载 Paraformer 模型 {model}（首次会下载，要几分钟）")
        self._model = AutoModel(model=model, disable_update=True)
        logger.debug(f"Paraformer 就绪，热词 {len(hotwords or [])} 个")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """转写一段语音。

        Args:
            audio: 16 kHz 单声道 16 位 PCM。

        Yields:
            ``TranscriptionFrame``，失败时 ``ErrorFrame``。
        """
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        def _transcribe() -> str:
            kwargs = {"hotword": self._hotword} if self._hotword else {}
            result = self._model.generate(input=samples, **kwargs)
            return (result[0]["text"] if result else "").strip()

        try:
            # 模型推理是同步的，扔线程里跑，别堵住事件循环。
            text = await asyncio.to_thread(_transcribe)
        except Exception as e:
            logger.error(f"{self} Paraformer 出错: {e}")
            await self.stop_processing_metrics()
            yield ErrorFrame(f"Paraformer transcription error: {e}")
            return

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            yield TranscriptionFrame(text, "", time_now_iso8601())
