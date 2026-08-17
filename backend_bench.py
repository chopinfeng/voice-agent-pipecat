"""比一比两个 agent 后端：延迟、轮数、token、答对率。

自带循环和 Claude Agent SDK 干的是同一件事，代价却不一样：SDK 那边每轮都要重发
Claude Code 的系统提示和全套工具 schema，输入 token 高一个数量级（绝大部分是缓存读，
价钱只有正常输入的十分之一，所以别只看总数）。两边都指到 OpenRouter 跑同一个模型，
比的是**框架开销**而不是模型差异。

单次采样说明不了问题——同一个问题两次跑能差出一倍墙钟，答案也时好时坏。所以每个
问题跑 ``REPEAT`` 轮，报中位数和答对率。有确定答案的问题写上 ``expect``，直接核对，
不然「更快」可能只是更快地答错。

运行：
    uv run --project pipecat python backend_bench.py
    REPEAT=3 BENCH_MODEL=deepseek/deepseek-v4-flash uv run --project pipecat python backend_bench.py
"""

import asyncio
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

from openai import AsyncOpenAI

import backends


@dataclass
class Case:
    """一个测试问题，以及怎么算它答对了。"""

    question: str
    expect: tuple[str, ...] = ()
    reject: tuple[str, ...] = ()

    def graded(self, answer: str) -> bool | None:
        """答案对不对。没写判据的返回 None，不计入答对率。"""
        if not self.expect and not self.reject:
            return None
        return all(e in answer for e in self.expect) and not any(
            r in answer for r in self.reject
        )


# 现数而不是写死。写死过一次，加了个脚本之后判据就过期了，两个后端答的 28 明明是对的
# 却全判成错。
PY_COUNT = str(len(list(HERE.glob("*.py"))))

CASES = [
    # 简单题：一次工具调用就能答对，两个后端都是满分，用来兜住回归。
    # 「没有」是那次自相矛盾的答案的原话，留着当反例。
    Case("这个项目根目录下有几个 python 文件？", expect=(PY_COUNT,), reject=("没有",)),
    Case("filler.py 是干什么用的？", expect=("填充",)),
    # 难题：答案分散在注释和多个文件里，光看文件名或只读一个文件必然答不全。
    # 判据取的是「必须同时说到的几个点」——少一个就说明它没真读进去。
    Case(
        "后台任务并发满了之后，调度器有哪几种处理方式？",
        expect=("排队", "抢占", "挂起"),
    ),
    Case(
        "填充语为什么必须放在 TTS 之后，放前面会怎样？",
        expect=("上下文",),
    ),
    Case(
        "VAD 的 stop_secs 和轮次窗口分别决定什么？改哪个能治句子被切碎？",
        expect=("段", "轮"),
    ),
    Case("延迟测试是怎么做的？"),
]

GATEWAY = "https://openrouter.ai/api/v1"
REPEAT = int(os.getenv("REPEAT", "3"))


@dataclass
class Run:
    """一次运行的结果。"""

    secs: float
    steps: int
    inp: int
    cached: int
    out: int
    answer: str
    ok: bool | None


@dataclass
class Tally:
    """一个后端在所有运行上的汇总。"""

    runs: list[Run] = field(default_factory=list)

    def median(self, attr: str) -> float:
        return statistics.median(getattr(r, attr) for r in self.runs) if self.runs else 0

    def accuracy(self) -> tuple[int, int]:
        graded = [r for r in self.runs if r.ok is not None]
        return sum(r.ok for r in graded), len(graded)


class Counter:
    """记一次运行调了几次工具。"""

    def __init__(self):
        self.steps = 0

    async def on_step(self, step: int, note: str, reads_file: bool) -> None:
        self.steps += 1

    async def gate(self) -> None:
        pass


async def run_one(backend, case: Case) -> Run:
    """跑一个问题。"""
    c = Counter()
    t = time.time()
    try:
        answer = await backend.run(case.question, on_step=c.on_step, gate=c.gate)
    except Exception as e:  # noqa: BLE001 - 一边挂了不该拖垮整个对比
        answer = f"<失败 {type(e).__name__}: {e}>"
    u = backend.usage
    answer = answer.replace("\n", " ")
    return Run(
        time.time() - t, c.steps, u.input, u.cached, u.output, answer, case.graded(answer)
    )


async def main():
    key = os.environ["OPENROUTER_API_KEY"]
    model = os.getenv("BENCH_MODEL", "z-ai/glm-4.6")
    client = AsyncOpenAI(api_key=key, base_url=GATEWAY)

    os.environ.setdefault("CLAUDE_BASE_URL", GATEWAY)
    os.environ.setdefault("CLAUDE_AUTH_TOKEN", key)
    os.environ["CLAUDE_AGENT_MODEL"] = model

    lineup = {
        "builtin": backends.build("builtin", client=client, model=model, root=HERE),
        "claude": backends.build("claude", client=client, model=model, root=HERE),
    }
    tallies = {name: Tally() for name in lineup}

    print(f"\n模型：{model}，每题 {REPEAT} 轮（两边同一个模型，比的是框架开销）\n")
    for case in CASES:
        print(f"■ {case.question}")
        for name, backend in lineup.items():
            for i in range(REPEAT):
                r = await run_one(backend, case)
                tallies[name].runs.append(r)
                mark = {True: "对", False: "错", None: "  "}[r.ok]
                print(
                    f"   {name:<8} #{i + 1} {r.secs:5.1f}s {r.steps:>2} 步 "
                    f"入 {r.inp:>6} 缓存 {r.cached:>6} 出 {r.out:>4} "
                    f"{mark} {r.answer[:38]}"
                )
        print()

    print("中位数")
    for name, t in tallies.items():
        hit, total = t.accuracy()
        rate = f"{hit}/{total}" if total else "—"
        print(
            f"   {name:<8} {t.median('secs'):5.1f}s  {t.median('steps'):.0f} 步  "
            f"入 {t.median('inp'):>6.0f} 缓存 {t.median('cached'):>7.0f} "
            f"出 {t.median('out'):>4.0f}   答对 {rate}"
        )


if __name__ == "__main__":
    asyncio.run(main())
