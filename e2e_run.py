"""端到端场景跑一遍并自动判定。给人看，也给 CI 看。

跟 ``scenarios.py`` 的关系：那个负责**把场景跑起来**（合成语音、喂进真实管线、
记录时间线），这里负责**判断跑得对不对**。分开是因为判据会比场景改得频繁得多。

输出两份：终端上是带时间戳的时间线加逐条判定；``logs/e2e_report.json`` 是机器读的，
CI 拿它出报告、比历史。

退出码：全过 0，有失败 1，只有「跳过」（测量没成立）也是 0——**基础设施抖动不该
把 CI 染红**，那样红灯久了就没人看。

跑之前会做体检：机器负载太高、有残留的 voice_bot 进程，直接拒跑。这两样都会把延迟
放大几十倍（实测残留进程堆到负载 44 时，本地听写量出 68.9 秒）。

运行：
    uv run --project pipecat python e2e_run.py              # 全部
    uv run --project pipecat python e2e_run.py --tag core   # 只跑核心回归
    uv run --project pipecat python e2e_run.py 打断          # 名字匹配
    uv run --project pipecat python e2e_run.py --list
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
REPORT = HERE / "logs" / "e2e_report.json"
MAX_LOAD = float(os.getenv("E2E_MAX_LOAD", "6.0"))


def preflight() -> list[str]:
    """跑之前的体检。返回拦停的理由，空列表表示可以跑。

    这两条都是被坑过才加的：残留的 voice_bot 进程会一直占着 FunASR 模型，
    机器负载高的时候所有延迟判据都会失真——那种数据比没有更糟，因为它看起来像真的。
    """
    problems = []
    load1 = os.getloadavg()[0]
    if load1 > MAX_LOAD:
        problems.append(f"机器负载 {load1:.1f} 超过 {MAX_LOAD}，延迟判据会失真")
    stray = subprocess.run(
        ["pgrep", "-f", "voice_bot.py"], capture_output=True, text=True, check=False
    )
    if stray.stdout.strip():
        n = len(stray.stdout.strip().splitlines())
        problems.append(f"有 {n} 个残留的 voice_bot 进程，先 pkill -f voice_bot.py")
    return problems


def render(case, timeline, results, secs: float) -> dict:
    """打印一条场景的结果，并返回机器可读的那份。"""
    from checks import Verdict

    bad = [r for r in results if r.verdict is Verdict.FAIL]
    skipped = [r for r in results if r.verdict is Verdict.SKIP]
    mark = "✗ 失败" if bad else ("~ 跳过" if skipped else "✓ 通过")
    flag = "（易抖，不拦 CI）" if case.flaky and bad else ""
    print(f"\n■ {case.name}  {mark}{flag}  用时 {secs:.1f}s")

    marks = {"用户": ">>>", "听写": " ~ ", "助手": "   "}
    for at, who, text in sorted(timeline.lines, key=lambda r: r[0]):
        print(f"  [{at:6.1f}s] {marks.get(who, '   ')} {text[:76]}")
    for r in results:
        icon = {"通过": "✓", "失败": "✗", "跳过": "~"}[r.verdict.value]
        print(f"    {icon} {r.name}：{r.detail}")
    if bad and case.flaky:
        print(f"    ↳ 已知易抖：{case.flaky}")

    return {
        "name": case.name,
        "flaky": bool(case.flaky),
        "secs": round(secs, 1),
        "checks": [
            {"name": r.name, "verdict": r.verdict.value, "detail": r.detail}
            for r in results
        ],
        "timeline": [
            {"at": round(at, 2), "who": who, "text": text}
            for at, who, text in sorted(timeline.lines, key=lambda r: r[0])
        ],
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    import e2e_cases

    cases = e2e_cases.CASES
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
        cases = e2e_cases.by_tag(tag)
        args = [a for a in args if a != tag]
    if args:
        cases = [c for c in cases if any(a in c.name for a in args)]

    if "--list" in flags:
        for c in cases:
            print(f"  {c.name:<16} {c.total:>5.0f}s  {'/'.join(c.tags)}")
        return 0

    import checks as checks_mod
    from checks import Verdict, selftest

    # 判据自己先过关。判据写错了不会报错，只会安静地给出错误结论。
    problems = selftest()
    if problems:
        print("判据自检未通过，先修判据：")
        for p in problems:
            print("  ✗", p)
        return 1
    print(f"判据自检通过（{datetime.now():%H:%M:%S}）")

    for reason in preflight():
        print(f"体检未通过：{reason}")
        return 1

    from scenario_runner import run_case

    started = datetime.now()
    print(f"开始 {started:%Y-%m-%d %H:%M:%S}，{len(cases)} 条场景\n")

    rows, failed = [], 0
    for case in cases:
        t = time.time()
        load_before = os.getloadavg()[0]
        timeline = run_case(case)
        load_after = os.getloadavg()[0]
        # 存活判据强制置顶。没有它，助手一句话没说的场景会因为「没泄漏」这类判据
        # 无东西可查而真空通过——第一次冒烟就是这么绿的。
        # 跑之前查一次不够——环境是会中途垮的。实测有一轮开跑时负载 2.8，跑到一半
        # 别的程序把它顶到 148，本地听写从 0.7 秒变成 62 秒，判据全线失真。
        # 这种时候结果既不是「通过」也不是「失败」，是**没测成**。
        peak = max(load_before, load_after)
        if peak > MAX_LOAD:
            alive = checks_mod.Result(
                "测量环境",
                Verdict.SKIP,
                f"跑的过程中负载到了 {peak:.1f}（上限 {MAX_LOAD}），这轮不作数",
            )
        else:
            alive = checks_mod.alive(timeline)
        results = [alive]
        # 系统压根没动时，后面的判据全是在空数据上打转，报出来只会误导。
        if alive.verdict is Verdict.PASS:
            results += case.checks(timeline)
        rows.append(render(case, timeline, results, time.time() - t))
        if any(r.verdict is Verdict.FAIL for r in results) and not case.flaky:
            failed += 1

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "started": started.isoformat(timespec="seconds"),
                "finished": datetime.now().isoformat(timespec="seconds"),
                "failed": failed,
                "cases": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{'=' * 60}")
    print(f"{len(cases)} 条场景，{failed} 条失败（易抖的不计）。报告：{REPORT}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
