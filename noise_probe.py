"""在噪声下测真实 VAD 的行为，以及它给全链路带来的延迟。

之前的探针（``realtime_replay.py`` / ``vad_probe.py``）都是**手动注入**
``VADUserStartedSpeakingFrame`` 和 ``VADUserStoppedSpeakingFrame`` 的——VAD 分析器
虽然传进去了，却从没参与过判断。所以那些脚本量出来的「说完到出声」里根本不含 VAD
的判定耗时，改 ``stop_secs`` 也不会让数字动。这里换成 pipecat 的 ``VADProcessor``，
让 VAD 真的听音频、真的决定什么时候算说完。

安静环境下 VAD 判得准是理所当然的，真实房间不是这样。三种噪声各有各的坑：

* **白噪声**——空调、风扇。宽频但平稳，VAD 通常扛得住。
* **人声嘈杂（babble）**——最难的一种。旁边有人说话，频谱跟目标语音重合，
  VAD 分不出是谁在说，容易一直判成「还在说话」。
* **低频嗡鸣**——电流声、路噪。能量集中在低频，理论上好滤。

四个指标，前两个是延迟，后两个是正确性：

* 起判延迟：真的开口到 VAD 说「开始了」。
* 停判延迟：真的说完到 VAD 说「停了」。这才是 ``stop_secs`` 的实际代价。
* 分段数：一句话被切成几段听写。噪声会让 VAD 在句中误判停顿。
* **纯噪声误触发**：没人说话时 VAD 报了几次开口。这个最要命——它直接对应
  「助手无缘无故插话」。

运行：
    uv run --project pipecat python noise_probe.py
    SNRS=20,10 uv run --project pipecat python noise_probe.py
"""

import asyncio
import math
import os
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language

from filler import synthesize_clips
from filters import EmptyTranscriptionFilter, StripEmotionMarks

IN_WAV = HERE / "zh_test_input.wav"
MODEL_DIR = HERE / "piper-voices"
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))
VAD_CONFIDENCE = float(os.getenv("VAD_CONFIDENCE", "0.7"))
VAD_MIN_VOLUME = float(os.getenv("VAD_MIN_VOLUME", "0.6"))
CHUNK_SECS = 0.02
SNRS = [float(s) for s in os.getenv("SNRS", "99,20,10,5").split(",")]

# 旁人闲聊的台词，叠在一起当 babble 噪声。内容无所谓，要的是人声的频谱。
BABBLE_LINES = [
    "昨天那个会开到很晚才结束",
    "我觉得这个方案还可以再想想",
    "中午想吃点什么随便都行",
    "他下周应该会过来一趟",
]


class VADWatch(FrameProcessor):
    """记 VAD 什么时候说开始、什么时候说停止，以及听写出了几段。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.starts: list[float] = []
        self.stops: list[float] = []
        self.texts: list[str] = []
        # 时间基准取**第一帧音频到达**的时刻，不是构造时刻——FunASR 加载模型要好几秒，
        # 拿构造时刻当零点会把它算进 VAD 的延迟里。
        self.t0: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame) and self.t0 is None:
            self.t0 = time.perf_counter()
        now = time.perf_counter() - (self.t0 or time.perf_counter())
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self.starts.append(now)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.stops.append(now)
        await self.push_frame(frame, direction)


class TextSink(FrameProcessor):
    """收听写结果，记回同一个 ``VADWatch``。

    单独一个处理器是因为它得挂在 STT **后面**，而 VAD 事件和音频要在 STT
    **前面**看——听写结果不会往上游流。
    """

    def __init__(self, watch: VADWatch, **kwargs):
        super().__init__(**kwargs)
        self._watch = watch

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            self._watch.texts.append(frame.text.strip())
        await self.push_frame(frame, direction)


def load_speech() -> tuple[np.ndarray, int]:
    """读测试语音，转成 float32。"""
    with wave.open(str(IN_WAV), "rb") as wav:
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0, rate


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """线性重采样。做噪声用，不要求保真。"""
    if src == dst:
        return x
    idx = np.linspace(0, len(x) - 1, int(len(x) * dst / src))
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def babble(n: int, rate: int) -> np.ndarray:
    """把几句话错开叠起来，做成人声嘈杂噪声。

    Piper 的输出是 22050，测试音频是 24000，得先对齐采样率，否则叠出来的「人声」
    音高和语速都不对，当噪声用会偏离真实场景。
    """
    clips, clip_rate = synthesize_clips(
        download_dir=MODEL_DIR, voice=TTS_VOICE, phrases=BABBLE_LINES
    )
    mixed = np.zeros(n, dtype=np.float32)
    for i, clip in enumerate(clips):
        one = np.frombuffer(clip, dtype=np.int16).astype(np.float32) / 32768.0
        one = _resample(one, clip_rate, rate)
        # 每条各错开一点，且循环铺满，免得出现整段静音。
        tiled = np.tile(one, n // max(len(one), 1) + 2)
        offset = (i * rate) // 3
        mixed += tiled[offset : offset + n]
    return mixed / max(len(clips), 1)


def make_noise(kind: str, n: int, rate: int, rng: np.random.Generator) -> np.ndarray:
    """造一段指定类型的噪声。"""
    if kind == "白噪声":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "人声嘈杂":
        return babble(n, rate)
    if kind == "低频嗡鸣":
        t = np.arange(n) / rate
        hum = sum(np.sin(2 * math.pi * f * t) for f in (50, 100, 150, 200))
        return (hum / 4).astype(np.float32)
    raise ValueError(kind)


def mix(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """按指定信噪比把噪声混进语音。SNR 很大就等于不混。"""
    if snr_db >= 90:
        return speech
    s_rms = float(np.sqrt(np.mean(speech**2)))
    n_rms = float(np.sqrt(np.mean(noise**2))) or 1e-9
    gain = s_rms / (n_rms * (10 ** (snr_db / 20)))
    return np.clip(speech + noise * gain, -1.0, 1.0)


def to_frames(samples: np.ndarray, rate: int, lead: float, tail: float) -> list:
    """转成按 20 毫秒节奏送达的音频帧，前后各留一段只有噪声的静音。

    前导那段是用来看纯噪声会不会误触发的——VAD 在这段里报开口就是假阳性。
    """
    pcm = (samples * 32767).astype(np.int16).tobytes()
    chunk = int(rate * CHUNK_SECS) * 2
    out = []
    for i in range(0, len(pcm), chunk):
        out.append(
            InputAudioRawFrame(
                audio=pcm[i : i + chunk], sample_rate=rate, num_channels=1
            )
        )
        out.append(SleepFrame(sleep=CHUNK_SECS))
    return out


async def one_run(
    samples: np.ndarray,
    rate: int,
    lead_secs: float,
    confidence: float | None = None,
    min_volume: float | None = None,
) -> VADWatch:
    """跑一次：真实 VAD 判定 + 听写。不接 LLM，隔离出 VAD 这一段。

    Args:
        samples: 音频。
        rate: 采样率。
        lead_secs: 开头纯噪声的长度。
        confidence: VAD 置信阈值，None 用 pipecat 默认的 0.7。
        min_volume: VAD 音量门，None 用默认的 0.6。判定是两个条件**同时**满足，
            所以提高任一个都会让 VAD 更保守。
    """
    watch = VADWatch()
    params = VADParams(
        stop_secs=VAD_STOP_SECS,
        confidence=VAD_CONFIDENCE,
        min_volume=VAD_MIN_VOLUME,
    )
    if confidence is not None:
        params.confidence = confidence
    if min_volume is not None:
        params.min_volume = min_volume
    vad = VADProcessor(vad_analyzer=SileroVADAnalyzer(params=params))
    stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    frames = to_frames(samples, rate, lead_secs, 0.0)
    frames.append(SleepFrame(sleep=6.0))
    await run_test(
        Pipeline(
            [
                vad,
                watch,
                stt,
                StripEmotionMarks(),
                EmptyTranscriptionFilter(),
                TextSink(watch),
            ]
        ),
        frames_to_send=frames,
    )
    return watch


async def main():
    speech, rate = load_speech()
    rng = np.random.default_rng(20260813)
    lead = 1.5  # 开头的纯噪声，用来抓误触发
    speech_secs = len(speech) / rate

    padded = np.concatenate(
        [np.zeros(int(rate * lead), np.float32), speech, np.zeros(int(rate * 1.5), np.float32)]
    )
    speech_start, speech_end = lead, lead + speech_secs

    print(f"\n语音 {speech_secs:.2f} 秒，前后各留静音；stop_secs={VAD_STOP_SECS}")
    print("误触发 = 语音开始之前 VAD 报的开口次数（纯噪声段）\n")
    print(f"{'噪声':<10}{'SNR':>6}{'起判延迟':>10}{'停判延迟':>10}{'分段':>6}{'误触发':>8}  听写")

    async def report(kind: str, label: str, samples: np.ndarray) -> None:
        w = await one_run(samples, rate, lead)
        # 前导噪声段里报的开口都是假的。
        false_starts = sum(1 for s in w.starts if s < speech_start - 0.15)
        real_starts = [s for s in w.starts if s >= speech_start - 0.15]
        start_lag = (real_starts[0] - speech_start) if real_starts else float("nan")
        stop_lag = (w.stops[-1] - speech_end) if w.stops else float("nan")
        text = " | ".join(w.texts)[:30] or "（没听出来）"
        print(
            f"{kind:<10}{label:>6}{start_lag:9.2f}s{stop_lag:9.2f}s"
            f"{len(w.texts):6d}{false_starts:8d}  {text}"
        )

    # 安静档是共同基线，跟噪声类型无关，只跑一次。
    await report("（无）", "安静", padded)
    for kind in ("白噪声", "人声嘈杂", "低频嗡鸣"):
        noise_full = make_noise(kind, len(padded), rate, rng)
        for snr in (s for s in SNRS if s < 90):
            await report(kind, f"{snr:.0f}dB", mix(padded, noise_full, snr))

    if os.getenv("SWEEP") == "1":
        await sweep(padded, rate, lead, speech_start, speech_end)
    if os.getenv("GATE") == "1":
        await gate_check(padded, rate, lead)


async def gate_check(padded: np.ndarray, rate: int, lead: float) -> None:
    """看受话人判定能不能把 babble 混进来的杂话拦掉。

    VAD 那层已经证明治不了 babble（收紧阈值停判延迟一秒没省，还多切出杂话段），
    所以只能在听写之后判。这里把 ``AddresseeGate`` 挂进链路，逐条打印它的判定，
    看音量和关键词两层是不是真的分得开「用户对助手说的」和「旁人的话」。
    """
    from addressee import AddresseeGate

    rng = np.random.default_rng(20260813)
    print("\n\n受话人判定在 babble 下的表现（enforce 关，只观测）")
    print(f"{'场景':<14}{'判定':>10}{'音量':>8}{'分数':>7}  听写")

    # 用户说话 + 旁人背景；以及**用户根本没说话、只有旁人在聊**。后者才是这个门要挡的
    # 主场景：助手不该去回应它碰巧听见的对话。
    babble_only = make_noise("人声嘈杂", len(padded), rate, rng)
    cases = [
        ("混入 10dB", mix(padded, babble_only, 10.0)),
        ("混入 5dB", mix(padded, babble_only, 5.0)),
        ("只有旁人 近", babble_only * 0.5),
        ("只有旁人 远", babble_only * 0.12),
    ]
    for label, noisy in cases:
        gate = AddresseeGate(mode="off", log_path=HERE / "logs" / "_gate_noise.jsonl")
        watch = VADWatch()
        vad = VADProcessor(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))
        )
        stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
        frames = to_frames(noisy, rate, lead, 0.0)
        frames.append(SleepFrame(sleep=6.0))
        await run_test(
            Pipeline(
                [
                    vad,
                    stt,
                    StripEmotionMarks(),
                    EmptyTranscriptionFilter(),
                    gate,
                    TextSink(watch),
                ]
            ),
            frames_to_send=frames,
        )
        if not gate.records:
            print(f"{label:<14}{'—':>10}{'':>8}{'':>7}  （VAD 没触发，压根没进来）")
        for u in gate.records:
            print(f"{label:<14}{u.label:>10}{u.rms:8.3f}{u.score:7.2f}  {u.text[:32]}")


async def sweep(
    padded: np.ndarray,
    rate: int,
    lead: float,
    speech_start: float,
    speech_end: float,
) -> None:
    """扫 VAD 的两个阈值，看能不能治住 babble。

    ``speaking = confidence >= 阈值 且 volume >= min_volume``，两个条件是**与**的关系，
    提高任一个都会让 VAD 更保守。但保守过头会漏掉用户自己——所以每组参数都要在
    babble 和干净两种条件下各跑一遍，干净那档失守就直接出局。
    """
    rng = np.random.default_rng(20260813)
    noisy = mix(padded, make_noise("人声嘈杂", len(padded), rate, rng), 10.0)

    print("\n\nVAD 阈值扫描（人声嘈杂 10dB；干净那栏是不能失守的底线）")
    print(f"{'置信':>6}{'音量门':>8}{'停判延迟':>10}{'分段':>6}{'误触发':>8}{'干净起判':>10}  嘈杂下的听写")

    for conf in (0.7, 0.85, 0.95):
        for vol in (0.6, 0.7, 0.8):
            w = await one_run(noisy, rate, lead, conf, vol)
            false_starts = sum(1 for s in w.starts if s < speech_start - 0.15)
            stop_lag = (w.stops[-1] - speech_end) if w.stops else float("nan")

            clean = await one_run(padded, rate, lead, conf, vol)
            hits = [s for s in clean.starts if s >= speech_start - 0.15]
            clean_lag = (hits[0] - speech_start) if hits else float("nan")
            ok = "" if hits and clean.texts else "  ← 干净档失守"

            print(
                f"{conf:6.2f}{vol:8.2f}{stop_lag:9.2f}s{len(w.texts):6d}"
                f"{false_starts:8d}{clean_lag:9.2f}s  "
                f"{' | '.join(w.texts)[:26]}{ok}"
            )


if __name__ == "__main__":
    asyncio.run(main())
