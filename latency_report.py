"""汇总 logs/latency.jsonl，看各环节耗时的分布。

单轮的数字抖动很大（网络、模型都会），要判断优化有没有效果得看多轮的中位数和
p95。

运行：
    uv run --project pipecat python latency_report.py
    uv run --project pipecat python latency_report.py logs/别的实验.jsonl
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_LOG = HERE / "logs" / "latency.jsonl"


def quantile(values: list[float], q: float) -> float:
    """取分位数。样本太少时退化成最大值，好过报错。"""
    if len(values) < 2:
        return values[0]
    idx = min(int(round(q * (len(values) - 1))), len(values) - 1)
    return sorted(values)[idx]


def table(title: str, rows: dict[str, list[float]], by_median: bool = False):
    """打印一张统计表。

    Args:
        title: 表头。
        rows: 指标名 -> 样本列表。
        by_median: 按中位数升序排列。时间线用它，读出来就是事情发生的顺序。
    """
    if not rows:
        return
    items = [(k, v) for k, v in rows.items() if v]
    if by_median:
        items.sort(key=lambda kv: statistics.median(kv[1]))
    print(f"\n{title}")
    print(f"  {'指标':<22s}{'n':>4s}{'中位':>8s}{'均值':>8s}{'p95':>8s}{'最大':>8s}")
    for name, values in items:
        print(
            f"  {name:<22s}{len(values):>4d}{statistics.median(values):>7.2f}s"
            f"{statistics.fmean(values):>7.2f}s{quantile(values, 0.95):>7.2f}s"
            f"{max(values):>7.2f}s"
        )


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_LOG
    if not path.exists():
        sys.exit(f"没有找到 {path}，先跑一次 voice_bot.py 或 e2e_latency.py")

    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not records:
        sys.exit(f"{path} 是空的")

    spans: dict[str, list[float]] = {}
    steps: dict[str, list[float]] = {}
    ttfb: dict[str, list[float]] = {}
    ttfa: dict[str, list[float]] = {}
    for r in records:
        for k, v in r.get("spans_s", {}).items():
            spans.setdefault(k, []).append(v)
        for k, v in r.get("timeline_s", {}).items():
            steps.setdefault(k, []).append(v)
        # 处理器名带 #N 实例编号，每次进程重启都会变，按服务类型归并。
        for k, v in r.get("ttfb_s", {}).items():
            ttfb.setdefault(k.split("#")[0], []).append(v)
        for k, v in r.get("ttfa_s", {}).items():
            ttfa.setdefault(k.split("#")[0], []).append(v)

    print(f"{path}：{len(records)} 轮")
    table("关键区间", spans)
    table("各环节距轮次起点的累计时刻", steps, by_median=True)
    table("服务自报 TTFB", ttfb)
    table("服务自报 TTFA", ttfa)


if __name__ == "__main__":
    main()
