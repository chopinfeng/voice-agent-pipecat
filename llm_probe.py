"""拆解 LLM 首 token 延迟：网络握手、OpenRouter 路由、模型推理各占多少。

端到端里 LLM 那一段是大头，但「2.8 秒」是个混合数字。这里用三组对照把它拆开：

1. 裸 HTTP：同一个 httpx 连接连发几次，第一次含 TLS 握手，之后是纯往返。
2. 裸 OpenAI SDK：pipecat 底下用的就是它，看 SDK 层有没有额外开销。
3. pipecat 服务：同一个 service 实例连发几次，看连接有没有被复用。

运行：
    uv run --project pipecat python llm_probe.py
"""

import asyncio
import os
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.tests.utils import SleepFrame, run_test

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")
ROUNDS = 4

# 跟 voice_bot.py 一致，让测出来的数字可比。
SYSTEM = (
    "你是一个中文语音助手。你的回答会被朗读出来，所以不要使用表情符号、"
    "项目符号或任何无法朗读的格式，数字和单位都写成口语说法。"
    "回答控制在一到两句话以内，第一句尽量短，这样用户能更快听到声音。"
)
PROMPT = "用一句话介绍杭州。"


def body(model: str) -> dict:
    return {
        "model": model,
        "stream": True,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
    }


async def probe_http(model: str) -> list[float]:
    """裸 HTTP，同一个连接池连发几次。"""
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    out = []
    async with httpx.AsyncClient(headers=headers, timeout=60) as client:
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            async with client.stream("POST", f"{BASE_URL}/chat/completions", json=body(model)) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: ") and '"content"' in line:
                        break
            out.append(time.perf_counter() - t0)
    return out


async def probe_sdk(model: str) -> list[float]:
    """裸 OpenAI SDK，pipecat 底下用的就是它。"""
    client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url=BASE_URL)
    out = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        stream = await client.chat.completions.create(**body(model))
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                break
        out.append(time.perf_counter() - t0)
    await client.close()
    return out


class FirstToken(FrameProcessor):
    """记录一次生成里第一个文本帧到达的时刻。"""

    def __init__(self):
        super().__init__()
        self.at: float | None = None

    def arm(self):
        self.at = None

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if self.at is None and isinstance(frame, LLMTextFrame):
            self.at = time.perf_counter()
        await self.push_frame(frame, direction)


async def probe_pipecat(model: str) -> list[float]:
    """pipecat 的服务，同一个实例连发几次——看连接有没有被复用。"""
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=model,
            system_instruction=SYSTEM,
            extra={"extra_body": {"provider": {"sort": "latency"}}},
        ),
    )
    mark = FirstToken()
    out = []
    for _ in range(ROUNDS):
        context = LLMContext()
        context.add_message({"role": "user", "content": PROMPT})
        mark.arm()
        t0 = time.perf_counter()
        await run_test(
            Pipeline([llm, mark]),
            frames_to_send=[LLMContextFrame(context), SleepFrame(sleep=15.0)],
        )
        out.append((mark.at - t0) if mark.at else float("nan"))
    return out


def show(label: str, values: list[float]):
    cells = "  ".join(f"{v:5.2f}" for v in values)
    warm = values[1:]
    avg = sum(warm) / len(warm) if warm else float("nan")
    print(f"  {label:<32s}{cells}    热连接均值 {avg:5.2f}s")


CANDIDATES = [
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "openai/gpt-4.1-nano",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "deepseek/deepseek-chat-v3.1",
    "x-ai/grok-4-fast",
]


async def main():
    header = f"  {'方式':<32s}" + "  ".join(f"{'#' + str(i + 1):>5s}" for i in range(ROUNDS))

    print(f"一、同一条链路的三种调用方式（{MODEL}）\n")
    print(header)
    show("裸 HTTP", await probe_http(MODEL))
    show("裸 OpenAI SDK", await probe_sdk(MODEL))
    show("pipecat 服务", await probe_pipecat(MODEL))

    print(f"\n二、候选模型的首 token（裸 SDK，每个 {ROUNDS} 次）\n")
    print(header)
    for model in CANDIDATES:
        try:
            show(model, await probe_sdk(model))
        except Exception as e:  # noqa: BLE001 - 某个模型不可用不该中断整轮对比
            print(f"  {model:<32s}失败: {str(e)[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
