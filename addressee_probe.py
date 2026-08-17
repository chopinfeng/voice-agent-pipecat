"""测 AddresseeGate 到底分不分得开「对助手说」和「旁人闲聊」。

两组台词各自合成成语音，走完整的听写管线，看打分能不能把它们分到两边。关键看的
不是准确率有多高——这是三档方案里最粗的一档——而是**分数分布有没有重叠**。两组分数
完全搅在一起，说明这些信号没用，得直接上声学模型；分得开才值得继续往上做。

运行：
    uv run --project pipecat python addressee_probe.py
"""

import asyncio
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

from addressee import AddresseeGate
from filters import EmptyTranscriptionFilter, StripEmotionMarks, TermCorrection
from pipecat.frames.frames import (
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.transcriptions.language import Language
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

from scenarios import say_as_audio  # noqa: E402

CHUNK_SECS = 0.02

# 冲着助手说的
TO_ASSISTANT = [
    "帮我看一下这个项目的整体架构",
    "延迟测试那部分是怎么做的",
    "你能把刚才的结果再说一遍吗",
    "查一下依赖了哪些第三方库",
    "停一下，先别查了",
    "这个功能是怎么实现的",
]

# 旁边两个人在聊，助手不该接话
TO_HUMAN = [
    "你昨天看那个球赛了吗",
    "我觉得楼下那家咖啡还行吧",
    "他说下周要去出差呢",
    "哎呀这个天气真是够热的",
    "中午吃什么，还是老地方",
    "你觉得呢，我是无所谓啦",
]


async def score_all(lines: list[str]) -> list[float]:
    """把每句合成成语音喂进管线，收集打分。"""
    gate = AddresseeGate(log_path=HERE / "logs" / "addressee_probe.jsonl")
    stt = FunASRSTTService(settings=FunASRSTTService.Settings(language=Language.ZH))
    worker = PipelineWorker(
        Pipeline(
            [stt, StripEmotionMarks(), TermCorrection(), EmptyTranscriptionFilter(), gate]
        ),
        name="probe",
        params=PipelineParams(),
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    task = asyncio.create_task(runner.run())
    await asyncio.sleep(1.0)

    scores: list[float] = []
    heard: list[str] = []
    seen = 0
    for line in lines:
        pcm, rate = say_as_audio(line)
        pcm = pcm + b"\x00" * (int(rate * 0.8) * 2)
        chunk = int(rate * CHUNK_SECS) * 2
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
        await asyncio.sleep(2.5)

        # 直接读 gate 内存里的记录。一句话可能被切成多段产生多条，取这一句期间
        # 新增的里分最高的那条——多段里通常只有一段带实质内容。
        fresh = gate.records[seen:]
        seen = len(gate.records)
        scores.append(max((r.score for r in fresh), default=float("nan")))
        heard.append(" | ".join(r.text for r in fresh) or "(没听出来)")

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return scores, heard


async def main():
    (HERE / "logs" / "addressee_probe.jsonl").unlink(missing_ok=True)

    print("=== 对助手说的 ===")
    a, a_heard = await score_all(TO_ASSISTANT)
    for line, s, h in zip(TO_ASSISTANT, a, a_heard, strict=True):
        print(f"  {s:.2f}  {line}\n        听成：{h}")

    print("\n=== 旁人闲聊 ===")
    b, b_heard = await score_all(TO_HUMAN)
    for line, s, h in zip(TO_HUMAN, b, b_heard, strict=True):
        print(f"  {s:.2f}  {line}\n        听成：{h}")

    a = [x for x in a if x == x]
    b = [x for x in b if x == x]
    if not a or not b:
        print("\n没收集到足够分数")
        sys.exit(1)

    print(f"\n对助手  中位 {statistics.median(a):.2f}  范围 {min(a):.2f}-{max(a):.2f}")
    print(f"旁人聊  中位 {statistics.median(b):.2f}  范围 {min(b):.2f}-{max(b):.2f}")
    gap = min(a) - max(b)
    if gap > 0:
        print(f"\n两组完全分开，中间还空出 {gap:.2f}。阈值可以取 {max(b) + gap / 2:.2f}")
    else:
        overlap = [x for x in a if x <= max(b)] + [x for x in b if x >= min(a)]
        print(f"\n两组有重叠，{len(overlap)} 句落在交叠区——光靠这些信号分不干净")


if __name__ == "__main__":
    asyncio.run(main())
