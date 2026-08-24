"""量对话链路上 LLM 的首 token 延迟，挑最快的那个当对话模型。

端到端里 LLM 是最大的一笔，而它是**唯一可以换掉的**——STT 和 TTS 在本地跑，
VAD 那 0.8 秒是判断说完必须等的。所以对话延迟能不能再降，基本就看这里。

两件事一起量：

* **候选模型的首 token**，每个跑 ``ROUNDS`` 次热连接取中位。冷启动那次含 TLS 握手
  （实测能到一秒多），单独列出来但不计入中位。
* **带不带工具定义的差别**。语音侧挂着四个工具，schema 每轮都要重发；如果它显著拖慢
  首 token，那就值得把工具收窄到真正用得上的时候再挂。

对话模型和 agent 模型是分开配的（``OPENROUTER_MODEL`` / ``AGENT_MODEL``），这里只管
前者——它只需要说得快、中文自然，复杂活儿交给后台 agent。

运行：
    uv run --project pipecat python ttfb_probe.py
    ROUNDS=7 uv run --project pipecat python ttfb_probe.py
"""

import asyncio
import math
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

ROUNDS = int(os.getenv("ROUNDS", "5"))

# 跟 voice_bot.py 的对话提示保持一致，否则量出来的数跟线上对不上。
SYSTEM = (
    "你是一个中文语音助手。你的回答会被朗读出来，所以不要使用表情符号、"
    "项目符号或任何无法朗读的格式，数字和单位都写成口语说法。"
    "回答控制在一到两句话以内，第一句尽量短，这样用户能更快听到声音。"
)
PROMPT = "帮我看一下这个项目的延迟主要花在哪儿了。"

# 只用中国开源模型：这个 key 上 Anthropic / Google / OpenAI 全返回 403
# 「违反 provider 服务条款」，而且用户明确要求只用国产开源。
CANDIDATES = os.getenv(
    "CANDIDATES",
    "deepseek/deepseek-v4-flash,z-ai/glm-4.6,qwen/qwen3-max,"
    "minimax/minimax-m2.7,inclusionai/ling-3.0-flash,moonshotai/kimi-k2.5",
).split(",")

# 语音侧挂的工具，只取名字和形状，用来量 schema 对首 token 的影响。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    }
    for name, desc in (
        ("ask_project", "把一个需要翻代码才能回答的问题交给后台 agent"),
        ("ask_another", "后台已经在忙时，再排一个新问题"),
        ("check_progress", "查后台任务进行到哪一步了"),
        ("cancel_task", "取消后台任务"),
    )
]


async def ttfb(client, model: str, tools: list | None) -> float:
    """发一次流式请求，返回收到第一个内容 token 的秒数。"""
    t = time.time()
    stream = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        tools=tools or None,
        stream=True,
        max_tokens=int(os.getenv('MAX_TOKENS', '400')),
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and (delta.content or delta.tool_calls):
            await stream.close()
            return time.time() - t
    return float("nan")


async def measure(client, model: str, tools: list | None) -> tuple[float, float, float]:
    """跑一轮冷启动加 ROUNDS 轮热连接。

    Returns:
        (冷启动, 热连接中位, 热连接最大值)。**最大值必须报**——语音链路上决定体验的
        是尾巴不是中位数：上一轮选型时 deepseek 的中位只比对手慢 0.4 秒，p95 却是
        12 秒，隔几轮就卡一次，光看中位会选错。
    """
    try:
        cold = await ttfb(client, model, tools)
        warm = [await ttfb(client, model, tools) for _ in range(ROUNDS)]
    except Exception as e:  # noqa: BLE001 - 一个模型挂了不该中断整轮对比
        print(f"  {model} 失败：{type(e).__name__} {e}")
        return float("nan"), float("nan"), float("nan")
    return cold, statistics.median(warm), max(warm)


async def main():
    key = os.environ["OPENROUTER_API_KEY"]
    print(f"\n首 token 延迟，每个模型 {ROUNDS} 次热连接取中位\n")
    print(f"{'模型':<30}{'冷启动':>8}{'无工具':>8}{'带工具':>8}{'带工具最慢':>11}")

    for model in CANDIDATES:
        # 每个模型一个新客户端，保证冷启动那次是真的冷。
        client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        cold, bare, _ = await measure(client, model, None)
        _, withtools, worst = await measure(client, model, TOOLS)
        if math.isnan(bare):
            # 推理型模型（glm-4.6 这类）在 max_tokens 用完之前可能只吐 reasoning，
            # 一个正文 token 都没有。这种本来也不适合走对话链路。
            print(f"{model:<32}    没等到正文 token，不适合对话链路")
            continue
        print(
            f"{model:<30}{cold:7.2f}s{bare:7.2f}s{withtools:7.2f}s{worst:10.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
