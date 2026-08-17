"""比较本地 STT 候选的延迟和准确率。

接上真实 VAD 之后，听写是链路上仅次于 LLM 的一笔，而且它挡在轮次判定前面——轮次窗口
已经被它完全吸收，所以听写不提速，窗口调到 0 也没用。

**只快不准没有意义。**中文听错一个词，后面派给 agent 的问题就是错的（这个项目里
「填充语」被听成「春雨」害过一整轮测试）。所以每个候选同时量两件事：

* 延迟：段落闭合到听写结果返回，热跑取中位。
* 字错率（CER）：跟参考文本比的编辑距离除以参考长度，标点和空格都去掉再比。

候选都是本地模型——云端 STT 再快也要一个往返，这条链路已经有一个 LLM 往返了。

注意这些台词是 Piper 合成的，比真人录音干净，所以 CER 会偏乐观；用来横向比较可以，
别当成真实环境的绝对值。

运行：
    uv run --project pipecat python stt_probe.py
"""

import asyncio
import math
import os
import statistics
import time
import wave
from pathlib import Path

import zhconv
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from filler import synthesize_clips

MODEL_DIR = HERE / "piper-voices"
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")
REAL_WAV = HERE / "zh_test_input.wav"

# 台词覆盖几类实际会说的话：日常提问、项目术语、数字、稍长的复合句。
LINES = [
    "你好，请用一句话介绍一下杭州这座城市",
    "帮我看一下这个项目的整体架构",
    "延迟测试那部分是怎么做的",
    "把结果压缩成三句话",
    "现在后台有几个任务在跑",
]

_PUNCT = "，。？！、；：,.?!;: \n　"


def norm(s: str) -> str:
    """去掉标点空格、繁体转简体，再比。

    繁简必须归一化，否则数字会完全失真：whisper 系列默认吐繁体，「現在後台有幾個任務
    在跑」内容一个字没错，逐字比却是满错，第一版就这么把它的字错率算成了 29%。
    标点同理——听写加不加逗号不该算错。
    """
    s = zhconv.convert(s, "zh-cn")
    return "".join(c for c in s if c not in _PUNCT)


def cer(ref: str, hyp: str) -> float:
    """字错率：编辑距离除以参考长度。"""
    ref, hyp = norm(ref), norm(hyp)
    if not ref:
        return 0.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(
                prev[j - 1] if r == h else 1 + min(prev[j - 1], prev[j], cur[j - 1])
            )
        prev = cur
    return prev[-1] / len(ref)


class Catch(FrameProcessor):
    """记段落闭合到听写返回的耗时，以及听写文本。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.closed_at: float | None = None
        self.text = ""
        self.secs = float("nan")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStoppedSpeakingFrame):
            self.closed_at = time.perf_counter()
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            if self.closed_at and self.text == "":
                self.secs = time.perf_counter() - self.closed_at
            self.text += frame.text.strip()
        await self.push_frame(frame, direction)


def clips() -> list[tuple[str, bytes, int]]:
    """测试音频：合成的台词，加上仓库里那段真实录音。"""
    audio, rate = synthesize_clips(
        download_dir=MODEL_DIR, voice=TTS_VOICE, phrases=LINES
    )
    out = [(line, pcm, rate) for line, pcm in zip(LINES, audio, strict=True)]
    with wave.open(str(REAL_WAV), "rb") as wav:
        out.append(
            ("你好，请用一句话介绍一下杭州这座城市", wav.readframes(wav.getnframes()),
             wav.getframerate())
        )
    return out


async def transcribe(make_stt, pcm: bytes, rate: int) -> tuple[float, str]:
    """喂一段音频给听写，返回 (耗时, 文本)。

    VAD 帧在这里是**手动注入**的，因为要隔离的就是听写本身——真实 VAD 的判定耗时
    另有 noise_probe.py 量。段落边界给死了，各候选拿到的输入完全一样。
    """
    catch = Catch()
    chunk = int(rate * 0.02) * 2
    frames = [VADUserStartedSpeakingFrame()]
    for i in range(0, len(pcm), chunk):
        frames.append(
            InputAudioRawFrame(audio=pcm[i : i + chunk], sample_rate=rate, num_channels=1)
        )
    frames += [VADUserStoppedSpeakingFrame(), SleepFrame(sleep=12.0)]
    # 观察器挂在听写**后面**：听写结果不会往上游流，而 VAD 帧是往下游穿过去的，
    # 所以放后面两样都收得到。
    await run_test(Pipeline([make_stt(), catch]), frames_to_send=frames)
    return catch.secs, catch.text


def candidates() -> dict:
    """名字到构造函数。构造推迟到用的时候，免得没用上的模型也被下载。"""
    from pipecat.services.funasr.stt import FunASRSTTService
    from pipecat.services.whisper.stt import MLXModel, WhisperSTTServiceMLX

    return {
        "FunASR SenseVoiceSmall": lambda: FunASRSTTService(
            settings=FunASRSTTService.Settings(language=Language.ZH)
        ),
        "MLX whisper turbo-q4": lambda: WhisperSTTServiceMLX(
            settings=WhisperSTTServiceMLX.Settings(
                model=MLXModel.LARGE_V3_TURBO_Q4.value, language=Language.ZH
            )
        ),
        "MLX whisper tiny": lambda: WhisperSTTServiceMLX(
            settings=WhisperSTTServiceMLX.Settings(
                model=MLXModel.TINY.value, language=Language.ZH
            )
        ),
    }


async def main():
    data = clips()
    print(f"\n{len(data)} 段音频（{len(LINES)} 段合成 + 1 段真实录音）\n")
    print(f"{'候选':<26}{'延迟中位':>10}{'p95':>8}{'字错率':>9}  最差的一条")

    for name, make in candidates().items():
        # 第一次跑含模型加载，不计入。
        await transcribe(make, data[0][1], data[0][2])

        secs, errs, worst = [], [], ("", "", 0.0)
        for ref, pcm, rate in data:
            t, text = await transcribe(make, pcm, rate)
            e = cer(ref, text)
            secs.append(t)
            errs.append(e)
            if e >= worst[2]:
                worst = (ref, text, e)
        good = [s for s in secs if not math.isnan(s)]
        print(
            f"{name:<26}{statistics.median(good):9.2f}s{max(good):7.2f}s"
            f"{statistics.mean(errs):8.1%}  {worst[1][:22] or '（空）'}"
        )


if __name__ == "__main__":
    asyncio.run(main())
