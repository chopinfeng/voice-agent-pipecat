"""用随机留白压测轮次判定，给停顿窗口定参。

真人说长句时会在逗号、换气、想词的地方停一下，停多久完全随机。这些留白一旦超过
停顿窗口，一句话就会被判成好几轮：模型收到的是「把这个项目里。」「所有文件。」
这样的碎片，而不是完整的一句。

这里把台词按标点切开，在每个断点插入随机长度的静音，再按 20 毫秒一帧喂进听写
管线，统计一句话被切成了几段听写、几轮对话。

两个旋钮管两件不同的事，实测才分清的：

* VAD 的 ``stop_secs`` 决定**听写切几段**——静音一超过它，VAD 就判定说话停止，
  SegmentedSTTService 立刻切段去转写。
* 轮次策略的 ``user_speech_timeout`` 决定**这些段攒成几轮**交给大模型。

所以句子被切碎要调前者，调后者没用。两个开大都会抬高延迟，代价一起量。

随机种子固定，同一组参数每次跑出来一样，改了代码可以直接对比。

运行：
    uv run --project pipecat python vad_probe.py            # 扫一遍候选窗口
    uv run --project pipecat python vad_probe.py 0.4 0.8    # 只测这两个
"""

import asyncio
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from filler import synthesize_clips
from filters import EmptyTranscriptionFilter, StripEmotionMarks, TermCorrection

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

import voice_bot as V  # noqa: E402

CHUNK_SECS = 0.02
SEED = 20260808
# (VAD stop_secs, 轮次窗口)。第一组是当前默认值。
CANDIDATES = [(0.2, 0.4), (0.5, 0.6), (0.8, 0.8), (1.2, 1.2)]

# 几句典型的长台词，断点多、长短不一。
LINES = [
    "把这个项目里所有的代码文件都看一遍，说说整体架构是怎么组织的",
    "我想知道，延迟测试那部分，到底是怎么量的，用了哪些工具",
    "先别管那个了，你觉得，语音助手最难做的地方在哪儿呢",
]
# 在这些字后面断句。
BREAKS = "，。？！、"


def with_random_pauses(text: str, rng: random.Random) -> tuple[bytes, int, list[float]]:
    """按标点切开台词，逐段合成，在断点插入随机静音。

    Args:
        text: 台词。
        rng: 随机源，外部固定种子。

    Returns:
        (拼好的 PCM, 采样率, 每个断点的静音秒数)。
    """
    pieces, current = [], ""
    for ch in text:
        current += ch
        if ch in BREAKS:
            pieces.append(current)
            current = ""
    if current:
        pieces.append(current)

    clips, rate = synthesize_clips(
        download_dir=V.MODEL_DIR, voice=V.TTS_VOICE, phrases=pieces
    )
    audio, pauses = b"", []
    for i, clip in enumerate(clips):
        audio += clip
        if i < len(clips) - 1:
            # 0.15 到 1.0 秒：涵盖从换气到"想一下再说"的整个范围。
            gap = rng.uniform(0.15, 1.0)
            pauses.append(round(gap, 2))
            audio += b"\x00" * (int(rate * gap) * 2)
    audio += b"\x00" * (int(rate * 1.0) * 2)  # 收尾静音
    return audio, rate, pauses


class Counter(FrameProcessor):
    """数一句台词产生了几段听写、几次轮次结束。"""

    def __init__(self):
        super().__init__()
        self.transcripts: list[str] = []
        self.turns = 0
        self.first_at: float | None = None
        self.last_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self.transcripts.append(frame.text.strip())
            now = time.perf_counter()
            self.first_at = self.first_at or now
            self.last_at = now
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self.turns += 1
        await self.push_frame(frame, direction)


async def probe(stop_secs: float, timeout: float, rng: random.Random) -> dict:
    """用一组台词压一遍指定参数，返回统计。"""
    stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    counter = Counter()
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=stop_secs)),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=timeout)]
        ),
    )
    user_aggregator, _ = LLMContextAggregatorPair(
        LLMContext(), user_params=user_params
    )
    worker = PipelineWorker(
        Pipeline(
            [
                stt,
                StripEmotionMarks(),
                TermCorrection(),
                EmptyTranscriptionFilter(),
                counter,
                user_aggregator,
            ]
        ),
        name="probe",
        params=PipelineParams(),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.sleep(1.0)

    stats = []
    for line in LINES:
        counter.transcripts.clear()
        counter.turns = 0
        counter.first_at = counter.last_at = None

        pcm, rate, pauses = with_random_pauses(line, rng)
        chunk = int(rate * CHUNK_SECS) * 2
        spoke_at = time.perf_counter()
        await worker.queue_frames([VADUserStartedSpeakingFrame()])
        for i in range(0, len(pcm), chunk):
            await worker.queue_frames(
                [
                    InputAudioRawFrame(
                        audio=pcm[i : i + chunk], sample_rate=rate, num_channels=1
                    )
                ]
            )
            await asyncio.sleep(CHUNK_SECS)
        await worker.queue_frames([VADUserStoppedSpeakingFrame()])
        await asyncio.sleep(max(2.0, timeout + 1.5))

        audio_secs = len(pcm) / (rate * 2)
        stats.append(
            {
                "segments": len(counter.transcripts),
                "turns": counter.turns,
                "text": " | ".join(counter.transcripts),
                "pauses": pauses,
                # 说完到听写收齐：窗口开大了这个数就跟着涨。
                "settle": (counter.last_at - spoke_at - audio_secs)
                if counter.last_at
                else float("nan"),
            }
        )

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return {
        "stop_secs": stop_secs,
        "timeout": timeout,
        "avg_segments": sum(s["segments"] for s in stats) / len(stats),
        "avg_turns": sum(s["turns"] for s in stats) / len(stats),
        "avg_settle": sum(s["settle"] for s in stats) / len(stats),
        "detail": stats,
    }


async def main():
    if sys.argv[1:]:
        nums = [float(a) for a in sys.argv[1:]]
        wanted = list(zip(nums[::2], nums[1::2]))
    else:
        wanted = CANDIDATES
    print(f"随机种子 {SEED}，每组参数跑 {len(LINES)} 句带随机留白的长台词\n")
    print(
        f"{'VAD停顿':>8}{'轮次窗口':>10}{'平均分段':>10}"
        f"{'平均轮次':>10}{'说完到收齐':>12}"
    )
    print("-" * 52)
    results = []
    for stop_secs, timeout in wanted:
        # 每组用同一颗种子，留白序列完全一致，横向可比。
        r = await probe(stop_secs, timeout, random.Random(SEED))
        results.append(r)
        print(
            f"{stop_secs:>8.1f}{timeout:>10.1f}{r['avg_segments']:>10.1f}"
            f"{r['avg_turns']:>10.1f}{r['avg_settle']:>11.2f}s"
        )

    print("\n各句听写结果：")
    for r in results:
        print(f"\n  VAD停顿 {r['stop_secs']:.1f}s / 轮次窗口 {r['timeout']:.1f}s")
        for s in r["detail"]:
            print(f"    留白 {s['pauses']} -> {s['segments']} 段")
            print(f"      {s['text']}")


if __name__ == "__main__":
    asyncio.run(main())
