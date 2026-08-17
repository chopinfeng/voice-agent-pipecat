"""并发度性能测试：开大到底值不值。

`AGENT_CONCURRENCY` 默认值当初是拍的。开大了单个任务会不会被拖慢（几个工具循环
同时打 OpenRouter、同时读文件）？总吞吐能不能真的上去？名额满了要问 LLM 一次，
那笔开销占多少？这些都得量。

不走语音，直接往 agent worker 派 job——要测的是后台侧的吞吐，语音那一段是固定
成本，掺进来只会加噪声。

同一批任务在每个并发度下各跑一遍，任务内容固定，所以横向可比。注意每轮都真打
OpenRouter，跑一次的花费不算小。

**单次测量噪声很大**：同一档并发两次跑出来的总墙钟能差一倍（网络抖动、本地机器
负载都会掺进来）。想下结论就多跑几轮看分布，别拿一次的数字说事。

运行：
    uv run --project pipecat python perf_probe.py          # 1 / 2 / 4 各一轮
    uv run --project pipecat python perf_probe.py 1 3      # 只测这两档
    REPEAT=3 uv run --project pipecat python perf_probe.py # 每档跑三轮取中位
"""

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

import progress_api
from agent_worker import ProjectAgentWorker
from pipecat.workers.base_worker import BaseWorker
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

LEVELS = [1, 2, 4]
# 每档重复几轮。默认 1 轮够看趋势，要下结论就调大。
REPEAT = int(os.getenv("REPEAT", "1"))

# 轻重搭配：轻的能看出并发有没有把简单任务也拖慢，重的撑起吞吐差异。
TASKS = [
    "filler.py 是干什么的",
    "filters.py 里有哪几个过滤器，分别做什么",
    "progress_api.py 的进度状态有哪些字段",
    "延迟测试相关的代码有哪几个文件，分别负责什么",
    "scheduler.py 的并发闸门和暂停机制是怎么实现的",
    "这个项目的整体架构是怎么组织的",
]


class Driver(BaseWorker):
    """一口气把所有任务派出去，等它们全部回来。"""

    def __init__(self, name: str, questions: list[str]):
        super().__init__(name)
        self._questions = questions
        self.results: list[dict] = []
        self.wall = 0.0
        self.done = asyncio.Event()

    async def start(self):
        await super().start()
        self.create_task(self._go(), "go")

    async def _one(self, question: str) -> dict:
        t0 = time.perf_counter()
        try:
            async with self.job(
                "agent", name="ask", payload={"question": question}, timeout=600
            ) as job:
                async for _ in job:
                    pass
            answer = (job.response or {}).get("answer", "")
            ok = bool(answer)
        except Exception as e:  # noqa: BLE001 - 单个任务失败不该中断整轮测量
            answer, ok = str(e), False
        return {
            "question": question,
            "secs": time.perf_counter() - t0,
            "ok": ok,
            "chars": len(answer),
        }

    def _split(self) -> tuple[list[float], list[float]]:
        """从进度中心取每个任务的排队时长和纯执行时长。"""
        queued = [t.queued for t in progress_api.REGISTRY.values() if t.running_at]
        ran = [t.ran for t in progress_api.REGISTRY.values() if t.running_at]
        return queued, ran

    async def _go(self):
        await asyncio.sleep(1.5)
        t0 = time.perf_counter()
        # 全部同时派出去，让闸门去决定实际并行几个。
        self.results = await asyncio.gather(
            *(self._one(q) for q in self._questions)
        )
        self.wall = time.perf_counter() - t0
        self.done.set()


async def measure(concurrency: int) -> dict:
    """在指定并发度下跑完整批任务。"""
    progress_api.REGISTRY.clear()
    agent = ProjectAgentWorker(
        "agent",
        root=Path(os.getenv("PROJECT_PATH", str(HERE))),
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.getenv("AGENT_MODEL", "deepseek/deepseek-v4-flash"),
        concurrency=concurrency,
    )
    driver = Driver("driver", TASKS)
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(driver, agent)
    task = asyncio.create_task(runner.run())

    await driver.done.wait()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    secs = [r["secs"] for r in driver.results]
    queued, ran = driver._split()
    return {
        "concurrency": concurrency,
        "wall": driver.wall,
        # 端到端含排队，纯执行不含——并发主要缩短的是前者，后者反而可能被拖慢。
        "median": statistics.median(secs),
        "queued": statistics.median(queued) if queued else 0.0,
        "ran": statistics.median(ran) if ran else 0.0,
        "slowest": max(secs),
        "ok": sum(1 for r in driver.results if r["ok"]),
        "total": len(driver.results),
        "detail": driver.results,
    }


async def main():
    levels = [int(a) for a in sys.argv[1:]] or LEVELS
    print(f"{len(TASKS)} 个任务，每档跑 {REPEAT} 轮\n")
    print(
        f"{'并发':>4}{'总墙钟':>10}{'端到端中位':>12}{'其中排队':>10}"
        f"{'纯执行':>9}{'成功':>8}{'吞吐/分钟':>11}"
    )
    print("-" * 74)

    results = []
    for level in levels:
        rounds = [await measure(level) for _ in range(REPEAT)]
        # 多轮取中位，单轮就是它自己。
        r = min(rounds, key=lambda x: abs(x["wall"] - statistics.median(
            [y["wall"] for y in rounds]
        )))
        r["rounds"] = len(rounds)
        r["wall_spread"] = (max(x["wall"] for x in rounds)
                            - min(x["wall"] for x in rounds))
        results.append(r)
        throughput = r["total"] / r["wall"] * 60
        print(
            f"{level:>4}{r['wall']:>9.1f}s{r['median']:>11.1f}s{r['queued']:>9.1f}s"
            f"{r['ran']:>8.1f}s{r['ok']:>5}/{r['total']:<2}{throughput:>11.1f}"
        )
        if REPEAT > 1:
            print(f"     （{REPEAT} 轮总墙钟极差 {r['wall_spread']:.1f}s）")

    base = results[0]
    print(f"\n以并发 {base['concurrency']} 为基准：")
    for r in results[1:]:
        speedup = base["wall"] / r["wall"] if r["wall"] else 0
        exec_ratio = r["ran"] / base["ran"] if base["ran"] else 0
        verdict = "拖慢" if exec_ratio > 1.15 else "基本不受影响"
        print(
            f"  并发 {r['concurrency']}：总吞吐快 {speedup:.2f} 倍，"
            f"单任务纯执行 {exec_ratio:.2f} 倍（{verdict}）"
        )


if __name__ == "__main__":
    asyncio.run(main())
