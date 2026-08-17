"""消融实验：定位 LLM 那一段里「多出来的 1.3 秒」。

基线里 `LLM 请求发出 → 第一个 LLMTextFrame` 是 2.95 秒，但 LLM service 自报的
TTFB 只有 1.5 秒，裸管线打同样的请求也是 1.5 秒。多出来的部分只在完整管线里出现。

这里一次只改一个变量，看那个差值什么时候消失：

- 基线            完整管线，停顿窗口 0.4 秒（听写一回来就发请求）
- 请求延后        停顿窗口 3 秒，让 CPU 先空下来再发请求
- 去掉 TTS        管线里不放 TTS，排除合成侧的干扰
- 去掉 STT        跳过听写，直接把文本喂给聚合器

运行：
    uv run --project pipecat python latency_ablation.py [每组轮数]
"""

import asyncio
import os
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv
from latency_observer import LatencyObserver
from realtime_replay import realtime_frames

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    InputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.tests.utils import run_test
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

MODEL_DIR = HERE / "piper-voices"
LOG_DIR = HERE / "logs" / "ablation"
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")

# 没有 TTS 的变体收不到 TTSStoppedFrame，用 LLM 输出完毕收尾。
from pipecat.frames.frames import LLMFullResponseEndFrame  # noqa: E402


class StubTTSService(TTSService):
    """只吐静音的假 TTS，用来区分「TTSService 这个处理器」和「Piper 的 onnx 推理」。"""

    def __init__(self, **kwargs):
        super().__init__(
            settings=TTSSettings(model=None, voice=None, language=None), **kwargs
        )

    async def run_tts(self, text: str, context_id: str):
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(
            audio=b"\x00" * 3200, sample_rate=self.sample_rate, num_channels=1
        )
        yield TTSStoppedFrame()


class Dropper(FrameProcessor):
    """丢掉指定类型的高频帧，不让它们继续往下游走。

    一轮 4.5 秒的语音会产生 225 个 ``InputAudioRawFrame`` 和 322 个
    ``UserSpeakingFrame``。后者是 SystemFrame，走高优先级路径会插队，而下游的
    LLM 和 TTS 都用不到它们。
    """

    def __init__(self, kinds: tuple[type, ...]):
        super().__init__()
        self._kinds = kinds

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, self._kinds):
            return
        await self.push_frame(frame, direction)


VARIANTS = [
    ("基线", dict(speech_timeout=0.4, drop=())),
    ("丢 InputAudioRawFrame", dict(speech_timeout=0.4, drop=(InputAudioRawFrame,))),
    ("丢 UserSpeakingFrame", dict(speech_timeout=0.4, drop=(UserSpeakingFrame,))),
    ("两个都丢", dict(speech_timeout=0.4, drop=(InputAudioRawFrame, UserSpeakingFrame))),
]


async def one_turn(
    observer, *, speech_timeout: float, drop: tuple = (), tts: str = "piper"
):
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=LLM_MODEL,
            system_instruction="你是一个中文语音助手。回答控制在一到两句话以内，第一句尽量短。",
            extra={"extra_body": {"provider": {"sort": "latency"}}},
        ),
    )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext(),
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=speech_timeout
                    )
                ]
            ),
        ),
    )

    stages = [
        FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    ]
    stages.append(user_aggregator)
    if drop:
        stages.append(Dropper(drop))
    stages.append(llm)
    if tts == "piper":
        stages.append(
            PiperTTSService(
                download_dir=MODEL_DIR,
                settings=PiperTTSService.Settings(voice=TTS_VOICE),
            )
        )
    elif tts == "stub":
        stages.append(StubTTSService())
    stages.append(assistant_aggregator)

    await run_test(
        Pipeline(stages),
        observers=[observer],
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        frames_to_send=realtime_frames(),
    )


def summarize(path: Path) -> tuple[float, float, float]:
    """返回 (时间线上的 LLM 段, 服务自报 TTFB, 差值) 的中位数。"""
    import json

    spans, ttfbs = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        r = json.loads(line)
        span = r["spans_s"].get("LLM 首 token")
        ttfb = next((v for k, v in r["ttfb_s"].items() if "OpenRouter" in k), None)
        if span is not None and ttfb is not None:
            spans.append(span)
            ttfbs.append(ttfb)
    if not spans:
        return float("nan"), float("nan"), float("nan")
    a, b = statistics.median(spans), statistics.median(ttfbs)
    return a, b, a - b


async def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("预热模型...")
    warm = LatencyObserver(
        log_path=LOG_DIR / "_warmup.jsonl", finish_on=(TTSStoppedFrame,)
    )
    await one_turn(warm, speech_timeout=0.4, tts="piper")

    results = []
    for label, cfg in VARIANTS:
        path = LOG_DIR / (label.split("（")[0].replace(" ", "_") + ".jsonl")
        path.unlink(missing_ok=True)
        finish = (
            (LLMFullResponseEndFrame,) if cfg.get("tts") == "none" else (TTSStoppedFrame,)
        )
        observer = LatencyObserver(log_path=path, finish_on=finish)
        for i in range(rounds):
            print(f"{label} 第 {i + 1}/{rounds} 轮...")
            await one_turn(observer, **cfg)
        results.append((label, *summarize(path)))

    print(f"\n{'变体':<24s}{'时间线 LLM 段':>14s}{'自报 TTFB':>12s}{'待解释差值':>12s}")
    print("-" * 64)
    for label, span, ttfb, gap in results:
        print(f"{label:<24s}{span:>13.2f}s{ttfb:>11.2f}s{gap:>11.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
