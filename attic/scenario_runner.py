"""把一条 ``E2ECase`` 跑起来，返回带时间戳的 ``Timeline``。

薄薄一层适配：场景怎么跑（合成语音、喂进真实的双 worker 管线、按时间戳记录）
``scenarios.Harness`` 已经做好了，这里只负责把它的记录翻译成判据认识的 ``Timeline``，
再顺手从日志里抓几个用文本看不出来的事件（打断、派单、结果投递）。

**默认走真语音**：台词用 Piper 合成再按 20 毫秒一帧喂进去，过完整的听写链路。
纯注入文本测不出 STT 听岔——而这个项目里「算了不用查了」被听岔导致场景整体作废
发生过不止一次。设 ``E2E_TEXT=1`` 可以切回文本快模式，只验编排。
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from checks import Timeline

HERE = Path(__file__).parent
LOG = HERE / "logs" / "e2e_bot.log"

# 日志里这些行对应的事件。文本时间线看不出「打断发生过没有」，但它恰恰是
# 这套系统最容易出错的地方，所以要单独抓。
EVENT_PATTERNS = {
    "打断": re.compile(r"broadcasting interruption"),
    "派单": re.compile(r"派单给后台 agent"),
    "联网": re.compile(r"联网查询："),
    "投递": re.compile(r"投递结果"),
    "保住半句": re.compile(r"被打断，先记下已说出口的部分"),
}


def _collect_events(since: float) -> list[tuple[float, str]]:
    """从 loguru 的日志里抓事件。

    时间戳用日志行自己的，转成相对场景开始的秒数——跟时间线对得上才能写出
    「第 12 秒打断之后必须再出声」这样的判据。
    """
    if not LOG.exists():
        return []
    out: list[tuple[float, str]] = []
    for line in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        for name, pat in EVENT_PATTERNS.items():
            if pat.search(line):
                stamp = line[:23]
                try:
                    t = time.mktime(time.strptime(stamp[:19], "%Y-%m-%d %H:%M:%S"))
                except ValueError:
                    continue
                if t >= since:
                    out.append((t - since, name))
    return sorted(out)


def run_case(case, timeout: float = 400.0) -> Timeline:
    """跑一条场景，返回时间线。

    **每条场景起一个子进程。**本来是同进程里新建 Harness，为的是上下文和任务表从头
    开始；但每个 Harness 会加载一份 FunASR（约 1GB），六条跑下来模型全堆在一个进程里
    不释放，越跑越慢——实测第四条时听写要 62 秒、第五条干脆一条都没出来（正常是
    0.7 秒）。子进程退出时模型跟着还给系统，顺便也把「某条卡死拖垮整轮」隔开了。

    Args:
        case: 要跑的场景。
        timeout: 单条上限。卡住的场景返回空时间线，由存活判据把它标成「跳过」。
    """
    payload = json.dumps({"name": case.name})
    proc = subprocess.run(
        [sys.executable, __file__, "--one", payload],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        cwd=str(HERE),
    )
    marker = "===TIMELINE==="
    if marker not in proc.stdout:
        return Timeline()
    raw = json.loads(proc.stdout.rsplit(marker, 1)[1])
    return Timeline(
        lines=[(r["at"], r["who"], r["text"]) for r in raw["lines"]],
        events=[(e["at"], e["name"]) for e in raw["events"]],
    )


async def _wait_for_reply(harness, since: float, timeout: float) -> float | None:
    """等助手在 ``since`` 之后开口，返回开口时刻。超时返回 None。"""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        for at, who, _ in harness.transcript.lines:
            if who == "助手" and at >= since:
                return at
        await asyncio.sleep(0.2)
    return None


async def _drive(harness, case) -> None:
    """按脚本让用户说话。

    时间点**跟着事件走，不写死秒数**。真实语音链路有七八秒往返（按真实节奏喂音频、
    VAD 判定、听写、LLM），照绝对秒数排的话，「第 4 秒打断」会落在第一句还没被听写
    出来之前——那不是打断，是自言自语，测出来的东西跟想测的完全无关（第一版就这么
    把两条正常行为判成了失败）。

    脚本里的触发器有两种：数字是**相对场景开始的秒数**；``"回应后+N"`` 表示等助手
    开口之后再等 N 秒——打断类的场景必须用后者，因为「打断」的定义就是在对方说话时
    插进去。
    """
    for trigger, text in case.script:
        if isinstance(trigger, str) and trigger.startswith("回应后"):
            extra = float(trigger.split("+")[1]) if "+" in trigger else 0.0
            mark = time.perf_counter() - harness.transcript.t0
            at = await _wait_for_reply(harness, mark, timeout=45.0)
            if at is None:
                # 等不到就直接说，让判据去暴露「它根本没回应」这件事，
                # 而不是在这里静默跳过。
                pass
            else:
                await asyncio.sleep(extra)
        else:
            now = time.perf_counter() - harness.transcript.t0
            if trigger > now:
                await asyncio.sleep(trigger - now)
        await harness.say(text)

    remaining = case.total - (time.perf_counter() - harness.transcript.t0)
    if remaining > 0:
        await asyncio.sleep(remaining)


async def _run(case) -> Timeline:
    import scenarios

    harness = scenarios.Harness(voice=os.getenv("E2E_TEXT") != "1")
    wall_start = time.time()
    await harness.start()
    try:
        await _drive(harness, case)
    finally:
        await harness.stop()

    tl = Timeline(lines=list(harness.transcript.lines))
    tl.events = _collect_events(wall_start)
    return tl


def _child() -> None:
    """子进程入口：跑一条场景，把时间线打到标准输出。"""
    import e2e_cases

    name = json.loads(sys.argv[2])["name"]
    case = next(c for c in e2e_cases.CASES if c.name == name)
    tl = asyncio.run(_run(case))
    print("===TIMELINE===")
    print(
        json.dumps(
            {
                "lines": [{"at": at, "who": w, "text": t} for at, w, t in tl.lines],
                "events": [{"at": at, "name": n} for at, n in tl.events],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    if "--one" in sys.argv:
        _child()
