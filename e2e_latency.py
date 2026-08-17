"""端到端延迟测试：用户说完 -> 助手第一个音频帧。

跑的是 voice_bot.py 里那条完整管线（STT -> 上下文聚合 -> LLM -> TTS），
只是把麦克风换成一段 WAV，所以不需要真人说话就能量出用户感知的延迟。

运行：
    uv run --project pipecat python e2e_latency.py
"""

import asyncio
import os
import time
import wave
from pathlib import Path

from dotenv import load_dotenv
from latency_observer import LatencyObserver
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

IN_WAV = HERE / "zh_test_input.wav"
OUT_WAV = HERE / "zh_reply.wav"
MODEL_DIR = HERE / "piper-voices"
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")

# 聚合器会消费掉听写帧，TTS 会消费掉 LLM 文本帧，所以每一段后面各挂一个探针，
# 而不是在管线尾部挂一个。
PROBES = {
    "说完": VADUserStoppedSpeakingFrame,
    "听写": TranscriptionFrame,
    "首token": LLMTextFrame,
    "出声": TTSAudioRawFrame,
}


class Trace(FrameProcessor):
    """记录每类帧第一次经过的时刻，并累积带文本的帧内容。"""

    def __init__(self):
        super().__init__()
        self.marks: dict[str, float] = {}
        self.texts: dict[str, str] = {}

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        for key, kind in PROBES.items():
            if isinstance(frame, kind):
                if key not in self.marks:
                    self.marks[key] = time.perf_counter()
                text = getattr(frame, "text", None)
                if text:
                    self.texts[key] = self.texts.get(key, "") + text
        await self.push_frame(frame, direction)


def audio_frames() -> list[InputAudioRawFrame]:
    with wave.open(str(IN_WAV), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        rate, channels = wav.getframerate(), wav.getnchannels()
    # 尾部补一段静音，让 VAD 和轮次判定拿到真实对话里那样的收尾。
    pcm += b"\x00" * (int(rate * 1.0) * 2 * channels)
    chunk = int(rate * 0.02) * 2 * channels
    return [
        InputAudioRawFrame(
            audio=pcm[i : i + chunk], sample_rate=rate, num_channels=channels
        )
        for i in range(0, len(pcm), chunk)
    ]


async def run_turn(
    frames: list[InputAudioRawFrame],
    turn_stop: str = "default",
    speech_timeout: float = 0.6,
    model: str | None = None,
):
    stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=model or LLM_MODEL,
            system_instruction="你是一个中文语音助手。回答控制在一到两句话以内，第一句尽量短。",
            extra={"extra_body": {"provider": {"sort": "latency"}}},
        ),
    )
    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice=TTS_VOICE),
    )
    user_params = LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    if turn_stop == "speech-timeout":
        user_params.user_turn_strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=speech_timeout)]
        )
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        LLMContext(), user_params=user_params
    )

    after_stt, after_llm, after_tts = Trace(), Trace(), Trace()
    # 离线回放没有输出 transport，收不到 BotStoppedSpeakingFrame，改用 TTS 合成
    # 完毕作为一轮的收尾信号。
    observer = LatencyObserver(finish_on=(TTSStoppedFrame,))
    out, _ = await run_test(
        Pipeline(
            [
                stt,
                after_stt,
                user_aggregator,
                llm,
                after_llm,
                tts,
                after_tts,
                assistant_aggregator,
            ]
        ),
        observers=[observer],
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            *frames,
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=30.0),
        ],
    )
    marks = {
        "说完": after_stt.marks.get("说完"),
        "听写": after_stt.marks.get("听写"),
        "首token": after_llm.marks.get("首token"),
        "出声": after_tts.marks.get("出声"),
    }
    texts = {
        "heard": after_stt.texts.get("听写", ""),
        "reply": after_llm.texts.get("首token", ""),
    }
    return marks, texts, list(out)


async def report(frames, label: str, save_audio: bool = False, **kwargs):
    m, texts, out = await run_turn(frames, **kwargs)
    audio = [f for f in out if isinstance(f, TTSAudioRawFrame)]
    assert audio, "没有生成音频"

    if save_audio:
        pcm = b"".join(f.audio for f in audio)
        with wave.open(str(OUT_WAV), "wb") as wav:
            wav.setnchannels(audio[0].num_channels)
            wav.setsampwidth(2)
            wav.setframerate(audio[0].sample_rate)
            wav.writeframes(pcm)

    def gap(a: str, b: str) -> str:
        return f"{m[b] - m[a]:5.2f}" if m.get(a) and m.get(b) else "    -"

    print(
        f"{label:<34s} {gap('说完', '听写')} {gap('听写', '首token')} "
        f"{gap('首token', '出声')} {gap('说完', '出声')}   {texts['reply'].strip()[:26]}"
    )


async def main():
    frames = audio_frames()

    print("预热模型...")
    await run_turn(frames, "speech-timeout")

    print(
        f"\n听到 : {(await run_turn(frames, 'speech-timeout'))[1]['heard'].strip()}\n"
    )
    print(f"{'配置':<32s} {'听写':>6s} {'LLM':>6s} {'TTS':>6s} {'端到端':>7s}   回答")
    print("-" * 100)

    await report(frames, "默认 smart-turn", turn_stop="default")
    for timeout in (0.6, 0.4):
        await report(
            frames,
            f"speech-timeout {timeout}s",
            turn_stop="speech-timeout",
            speech_timeout=timeout,
            save_audio=timeout == 0.4,
        )
    for model in ("openai/gpt-4.1-nano", "openai/gpt-4.1-mini"):
        await report(
            frames,
            f"0.4s + {model.split('/')[-1]}",
            turn_stop="speech-timeout",
            speech_timeout=0.4,
            model=model,
        )


if __name__ == "__main__":
    asyncio.run(main())
