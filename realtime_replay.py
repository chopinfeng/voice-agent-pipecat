"""按真实节奏回放音频，量一轮完整对话的延迟。

e2e_latency.py 是把整段音频瞬间灌进管线的，VAD 会在几毫秒里连跑几百次推理，
和 LLM 的流抢事件循环——测出来的 LLM 那一段会比实际偏大。这里改成每 20 毫秒
喂一帧，跟麦克风一样，所以数字可以当作真实对话的近似。

运行：
    uv run --project pipecat python realtime_replay.py [轮数]
"""

import asyncio
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import InputAudioRawFrame, TTSStoppedFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from filler import DEFAULT_PHRASES, FillerSpeech, synthesize_clips
from filters import EmptyTranscriptionFilter
from latency_observer import LatencyObserver

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

import voice_bot as V  # noqa: E402 - 要先加载 .env；工具定义从这里取，保证跟线上一致

IN_WAV = HERE / "zh_test_input.wav"
MODEL_DIR = HERE / "piper-voices"
LOG = HERE / "logs" / "realtime.jsonl"
# 直接用 voice_bot 的值，别在这儿另留一份默认——两边默认值不一致时，
# 量出来的是一个线上没人见过的配置（SPEECH_TIMEOUT 就这么坑过一次）。
LLM_MODEL = os.getenv("OPENROUTER_MODEL", V.LLM_MODEL)
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")
# 默认值跟 voice_bot.py 保持一致——这个脚本要测的是真正在跑的配置，
# 各留一套默认值只会量出一个上线之后没人见过的数。
SPEECH_TIMEOUT = float(os.getenv("SPEECH_TIMEOUT", "0.4"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))
FILLER_DELAY = float(os.getenv("FILLER_DELAY", "0.25"))
# 设 WITH_TOOLS=0 量裸调用，用来隔离工具 schema 本身的代价。
WITH_TOOLS = os.getenv("WITH_TOOLS", "1") != "0"
CHUNK_SECS = 0.02


_FILLER_CACHE = None


def _filler_clips():
    """填充语只合成一次，多轮之间复用。"""
    global _FILLER_CACHE
    if _FILLER_CACHE is None:
        _FILLER_CACHE = synthesize_clips(
            download_dir=MODEL_DIR, voice=TTS_VOICE, phrases=DEFAULT_PHRASES
        )
    return _FILLER_CACHE


def realtime_frames() -> list:
    """音频帧之间插入 20 毫秒的等待，模拟麦克风的到达节奏。"""
    with wave.open(str(IN_WAV), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        rate, channels = wav.getframerate(), wav.getnchannels()
    pcm += b"\x00" * (int(rate * 1.0) * 2 * channels)  # 尾部静音，给轮次判定收尾
    chunk = int(rate * CHUNK_SECS) * 2 * channels

    out = []
    for i in range(0, len(pcm), chunk):
        out.append(
            InputAudioRawFrame(
                audio=pcm[i : i + chunk], sample_rate=rate, num_channels=channels
            )
        )
        out.append(SleepFrame(sleep=CHUNK_SECS))
    out.append(SleepFrame(sleep=25.0))
    return out


async def one_turn(observer: LatencyObserver):
    if os.getenv("STT") == "streaming":
        from streaming_stt import FunASRStreamingSTTService

        stt = FunASRStreamingSTTService()
    else:
        stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=LLM_MODEL,
            system_instruction="你是一个中文语音助手。回答控制在一到两句话以内，第一句尽量短。",
            extra={"extra_body": {"provider": {"sort": "latency"}}},
        ),
    )
    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice=TTS_VOICE),
    )
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)]
        ),
    )
    # 工具要跟着挂上。语音侧常驻四个工具，schema 每轮都要重发，而各家为此付的首 token
    # 代价差得离谱——不挂工具就是在量一个线上根本不存在的配置，两个模型会假性打平。
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext(tools=V.ALL_TOOLS if WITH_TOOLS else None), user_params=user_params
    )
    # VAD 必须真的跑。之前这里是手动注入 VADUserStarted/StoppedSpeakingFrame 的，
    # 分析器传了却从不参与判断——量出来的「说完到出声」不含 VAD 的判定耗时，
    # 改 stop_secs 数字也不会动。
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))
    )
    stages = [vad, stt, EmptyTranscriptionFilter(), user_aggregator, llm, tts]
    if FILLER_DELAY > 0:
        clips, rate = _filler_clips()
        stages.append(FillerSpeech(clips=clips, sample_rate=rate, delay=FILLER_DELAY))
    stages.append(assistant_aggregator)
    await run_test(
        Pipeline(stages),
        observers=[observer],
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        frames_to_send=realtime_frames(),
    )


async def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    observer = LatencyObserver(log_path=LOG, finish_on=(TTSStoppedFrame,))

    print("预热模型...")
    await one_turn(
        LatencyObserver(
            log_path=HERE / "logs" / "_warmup.jsonl", finish_on=(TTSStoppedFrame,)
        )
    )

    for i in range(rounds):
        print(f"第 {i + 1}/{rounds} 轮...")
        await one_turn(observer)

    print(f"\n结果写入 {LOG}，用下面这条命令看汇总：")
    print(f"  uv run --project pipecat python latency_report.py {LOG}")


if __name__ == "__main__":
    asyncio.run(main())
