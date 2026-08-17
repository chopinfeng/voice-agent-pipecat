"""在 Claude Agent SDK 这个框架下比不同模型的**效果**。

框架已经定了（实测框架自身只占 0.03% 的耗时，换框架不影响速度也不影响效果），
剩下的质量杠杆就是模型。这里固定后端、只换模型，跑 ``backend_bench`` 里那几道
**跨文件难题**——简单题各家都满分，区分不出来。

判分沿用 ``backend_bench.Case``：写死必须同时说到的几个点，少一个就算没读进去。
同时记 token，按 OpenRouter 价目折成每问成本——效果优先不等于不看代价，要看到
多花的钱换回了多少。

运行：
    uv run --project pipecat python agent_model_bench.py
    MODELS=z-ai/glm-4.6,anthropic/claude-sonnet-4.5 uv run --project pipecat python agent_model_bench.py
"""

import asyncio
import os
import statistics
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

import backends
from backend_bench import Counter
from evalset import CASES, sandbox_root

GATEWAY = "https://openrouter.ai/api/v1"
# 跑挂了的特征。这类**不是模型答错**，是测量本身失败——算进准确率会把网关抖动
# 记在模型头上（实测一轮里 15 次有 6 次是这样，sonnet 的 compute 因此从 4/5 变 1/5）。
BROKEN = ("API Error", "Can't reach", "<失败", "rate limit", "Connection")
RETRY = int(os.getenv("EVAL_RETRY", "2"))
MODELS = os.getenv(
    "MODELS",
    "z-ai/glm-4.6,anthropic/claude-haiku-4.5,anthropic/claude-sonnet-4.5",
).split(",")
REPEAT = int(os.getenv("REPEAT", "1"))

# OpenRouter 价目（美元每百万 token），用来把 token 折成每问成本。
PRICES = {
    "z-ai/glm-4.6": (0.50, 2.00, 0.10),
    "anthropic/claude-haiku-4.5": (1.00, 5.00, 0.10),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00, 0.30),
    "deepseek/deepseek-v4-flash": (0.14, 0.28, 0.028),
    "openai/gpt-5.1": (1.25, 10.00, 0.125),
    "moonshotai/kimi-k2-thinking": (0.60, 2.50, 0.15),
    "qwen/qwen3-max": (0.78, 3.90, 0.156),
}

# 只跑有判据的题——没有判据的题看不出高下，白花钱。
KINDS = os.getenv("KINDS", "")
HARD = [
    c
    for c in CASES
    if (c.expect or c.reject) and (not KINDS or c.kind in KINDS.split(","))
]


def cost(model: str, inp: int, cached: int, out: int) -> float:
    """按价目把 token 折成美元。"""
    p_in, p_out, p_cache = PRICES.get(model, (1.0, 5.0, 0.1))
    return (inp * p_in + cached * p_cache + out * p_out) / 1e6


async def run_model(model: str) -> dict:
    """一个模型跑完全部难题。"""
    os.environ["CLAUDE_AGENT_MODEL"] = model
    # **不能让 agent 看见考卷。**evalset.py 就在项目目录里，题目和答案都在
    # 里面——实测 glm 直接答「从 evalset.py 中的测试用例可以看出…」，那一题的
    # 分数毫无意义。所以 agent 跑在一份剔掉评测文件的副本上。
    backend = backends.build("claude", client=None, model=model, root=sandbox_root())

    hits = total = steps = broken = 0
    secs, spend = [], 0.0
    per_kind: dict[str, list[int]] = {}
    for case in HARD:
        for _ in range(REPEAT):
            # 跑挂了就重试。重试完还挂就记成「无效」，不计入准确率——
            # 把网关抖动算成模型答错，比样本少更容易得出反的结论。
            for attempt in range(RETRY + 1):
                c = Counter()
                t = time.time()
                try:
                    answer = await backend.run(
                        case.question, on_step=c.on_step, gate=c.gate, kind=case.kind
                    )
                except Exception as e:  # noqa: BLE001 - 一个模型挂了不该中断整轮对比
                    answer = f"<失败 {type(e).__name__}>"
                if not any(b in answer for b in BROKEN):
                    break
                if attempt < RETRY:
                    await asyncio.sleep(3)
            secs.append(time.time() - t)
            steps += c.steps
            u = backend.usage
            spend += cost(model, u.input, u.cached, u.output)
            if any(b in answer for b in BROKEN):
                broken += 1
                print(f"    ~ [{case.kind:<8}] 测量失败，重试 {RETRY} 次仍不通，不计分")
                continue
            ok = case.graded(answer)
            if ok is not None:
                total += 1
                hits += bool(ok)
            mark = {True: "对", False: "错", None: "  "}[ok]
            if ok is not None:
                per_kind.setdefault(case.kind, [0, 0])
                per_kind[case.kind][1] += 1
                per_kind[case.kind][0] += bool(ok)
            print(
                f"    {mark} [{case.kind:<8}] {time.time() - t:5.1f}s {c.steps:>2}步  "
                f"{answer[:40]}"
            )
    n = len(HARD) * REPEAT
    return {
        "model": model,
        "acc": f"{hits}/{total}",
        "broken": broken,
        "per_kind": "  ".join(
            f"{k}:{v[0]}/{v[1]}" for k, v in sorted(per_kind.items())
        ),
        "secs": statistics.median(secs),
        "steps": steps / n,
        "cost": spend / n,
    }


async def main():
    os.environ.setdefault("CLAUDE_BASE_URL", GATEWAY)
    os.environ.setdefault("CLAUDE_AUTH_TOKEN", os.environ["OPENROUTER_API_KEY"])

    print(f"\n框架固定为 Claude Agent SDK，只换模型。{len(HARD)} 道难题 × {REPEAT} 轮\n")
    rows = []
    for model in MODELS:
        print(f"■ {model}")
        rows.append(await run_model(model.strip()))
        print()

    print(f"{'模型':<30}{'答对':>7}{'耗时中位':>10}{'步数':>7}{'每问成本':>11}{'':>6}")
    for r in rows:
        print(
            f"{r['model']:<30}{r['acc']:>7}{r['secs']:9.1f}s{r['steps']:7.1f}"
            f"  ${r['cost']:.4f}  无效{r['broken']:>2}   {r['per_kind']}"
        )


if __name__ == "__main__":
    asyncio.run(main())
