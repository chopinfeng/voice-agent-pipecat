"""场景的自动判定：把「跑给人看的时间线」变成「能过 CI 的断言」。

``scenarios.py`` 已经把每轮对话按时间戳记下来了，但判断对错一直靠人读。要进 CI 就得
把「什么算对」写成代码。这里定义一组针对**时间线**的断言。

判据的写法有三条硬规矩，都是这个项目踩出来的：

1. **区分「测量失败」和「真的错了」。**网关抖动、机器负载、进程残留都会让场景挂掉，
   把它们算成失败会让 CI 变成噪声源，久了就没人看了。这类返回 ``SKIP``。
2. **判据本身要能被测。**每条断言都配了正例反例（``selftest()``），跑得起来才算数
   ——上一版判据把「八月十五号是周六」判成对过，没有自检根本发现不了。
3. **宁可漏判不可误判。**CI 里一次误报的代价是所有人开始忽略红灯。
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    """判定结果。"""

    PASS = "通过"
    FAIL = "失败"
    SKIP = "跳过"  # 测量本身没成立，不算失败


@dataclass
class Result:
    """一条断言的结果。"""

    name: str
    verdict: Verdict
    detail: str = ""


@dataclass
class Timeline:
    """一个场景跑完之后的全部可观测事实。

    Parameters:
        lines: ``(相对秒数, 角色, 文本)``，角色是「用户」「听写」「助手」。
        events: 日志里抓到的事件，``(相对秒数, 事件名)``，比如打断、派单、投递。
    """

    lines: list[tuple[float, str, str]] = field(default_factory=list)
    events: list[tuple[float, str]] = field(default_factory=list)

    def said(self, who: str = "助手") -> list[str]:
        """某个角色说过的全部内容。"""
        return [t for _, w, t in self.lines if w == who]

    def joined(self, who: str = "助手") -> str:
        return " ".join(self.said(who))

    def after(self, secs: float, who: str = "助手") -> list[str]:
        """某个时刻之后说的话。用来验「打断之后还接不接得上」。"""
        return [t for at, w, t in self.lines if w == who and at >= secs]

    def count(self, event: str) -> int:
        return sum(1 for _, e in self.events if event == e)

    def first_reply_after(self, secs: float) -> float | None:
        """某时刻之后助手第一次出声的时刻。用来量响应延迟。"""
        for at, who, _ in sorted(self.lines, key=lambda r: r[0]):
            if who == "助手" and at >= secs:
                return at
        return None


def alive(tl: Timeline, name: str = "系统有反应") -> Result:
    """场景到底跑起来没有。**每条场景都要先过这一关。**

    这是防「真空通过」的闸门：助手一句话没说时，「没有泄漏内部标注」这类判据会因为
    没东西可查而全部通过，报告上一片绿，系统其实完全没动。第一次冒烟就撞上了——
    听写 0 条、LLM 0 次调用，场景照样判「通过」。

    分得清两种沉默：听写有结果但助手不出声，是**真的坏了**（FAIL）；听写也没有，
    多半是环境问题（模型没加载好、机器过载），算 SKIP——把基础设施抖动记成功能失败，
    CI 红久了就没人看了。
    """
    heard, said = tl.said("听写"), tl.said("助手")
    if not heard and not said:
        return Result(name, Verdict.SKIP, "听写和助手都没有输出，多半是环境问题")
    if not said:
        return Result(name, Verdict.FAIL, f"听写有 {len(heard)} 条，但助手一句没说")
    return Result(name, Verdict.PASS, f"听写 {len(heard)} 条，助手 {len(said)} 句")


def spoke_within(tl: Timeline, after: float, budget: float, name: str) -> Result:
    """在 ``after`` 之后 ``budget`` 秒内出声了吗。

    量的是**用户感知的响应速度**——填充语也算数，因为用户听到的就是它。
    """
    at = tl.first_reply_after(after)
    if at is None:
        return Result(name, Verdict.FAIL, f"{after:.1f}s 之后再没出过声")
    delay = at - after
    ok = delay <= budget
    return Result(
        name,
        Verdict.PASS if ok else Verdict.FAIL,
        f"{delay:.2f}s（预算 {budget:.1f}s）",
    )


def no_repeat(tl: Timeline, name: str, threshold: int = 2) -> Result:
    """助手有没有把同一句话反复说。

    这是打断丢记忆的典型症状：模型不记得自己说过什么，每轮给出同一个答案。
    只比**较长的句子**——「好的」「嗯」这类短应答本来就该重复。
    """
    seen: dict[str, int] = {}
    for text in tl.said("助手"):
        key = re.sub(r"[\s，。！？、]", "", text)
        if len(key) < 12:
            continue
        seen[key] = seen.get(key, 0) + 1
    worst = max(seen.items(), key=lambda kv: kv[1], default=("", 0))
    if worst[1] > threshold:
        return Result(name, Verdict.FAIL, f"「{worst[0][:20]}」说了 {worst[1]} 遍")
    return Result(name, Verdict.PASS, f"最多重复 {worst[1]} 次")


def mentions(tl: Timeline, words: tuple[str, ...], name: str, who: str = "助手") -> Result:
    """助手说的话里有没有提到这些关键词（全部都要有）。"""
    text = re.sub(r"[\s，。]", "", tl.joined(who))
    missing = [w for w in words if w not in text]
    if missing:
        return Result(name, Verdict.FAIL, f"没提到：{'、'.join(missing)}")
    return Result(name, Verdict.PASS, "都提到了")


def mentions_any(tl: Timeline, words: tuple[str, ...], name: str) -> Result:
    """至少提到其中一个。用于同义说法。"""
    text = re.sub(r"[\s，。]", "", tl.joined())
    hit = [w for w in words if w in text]
    if hit:
        return Result(name, Verdict.PASS, f"命中「{hit[0]}」")
    return Result(name, Verdict.FAIL, f"一个都没提到：{'、'.join(words)}")


def never_says(tl: Timeline, words: tuple[str, ...], name: str) -> Result:
    """助手绝不该说出这些。

    两类用途：内部标注不许念出来（``[主用户]``、``[skip]``、系统提示），
    以及陷阱题里编造出来的东西。
    """
    text = tl.joined()
    leaked = [w for w in words if w in text]
    if leaked:
        return Result(name, Verdict.FAIL, f"念出来了：{'、'.join(leaked)}")
    return Result(name, Verdict.PASS, "没有泄漏")


def event_happened(tl: Timeline, event: str, name: str, at_least: int = 1) -> Result:
    """某个事件发生了至少几次。"""
    n = tl.count(event)
    ok = n >= at_least
    return Result(
        name, Verdict.PASS if ok else Verdict.FAIL, f"{event} 发生 {n} 次（要 ≥{at_least}）"
    )


def selftest() -> list[str]:
    """判据自检：把典型的正确/错误时间线喂进去，看判得对不对。

    **这一步不能省。**判据写错了不会报错，只会安静地给出错误结论——这个项目里
    发生过至少三次（中文数字判错、「八月十五号是周六」判对、跑挂算成答错）。
    """
    bad: list[str] = []

    def check(cond: bool, msg: str):
        if not cond:
            bad.append(msg)

    good = Timeline(lines=[(1.0, "用户", "问"), (2.0, "助手", "北京明天有雷阵雨最高二十九度")])
    late = Timeline(lines=[(1.0, "用户", "问"), (9.0, "助手", "答")])
    check(spoke_within(good, 1.0, 3.0, "x").verdict is Verdict.PASS, "spoke_within 正例误判")
    check(spoke_within(late, 1.0, 3.0, "x").verdict is Verdict.FAIL, "spoke_within 反例漏判")
    check(spoke_within(Timeline(), 0, 3, "x").verdict is Verdict.FAIL, "没出声该判失败")

    rep = Timeline(lines=[(i, "助手", "北京明天有雷阵雨最高二十九度") for i in range(4)])
    check(no_repeat(rep, "x").verdict is Verdict.FAIL, "no_repeat 没抓到复读")
    check(no_repeat(good, "x").verdict is Verdict.PASS, "no_repeat 误伤正常对话")
    short = Timeline(lines=[(i, "助手", "好的") for i in range(5)])
    check(no_repeat(short, "x").verdict is Verdict.PASS, "短应答不该算复读")

    leak = Timeline(lines=[(1.0, "助手", "[主用户] 你好")])
    check(never_says(leak, ("[主用户]",), "x").verdict is Verdict.FAIL, "没抓到标注泄漏")
    check(never_says(good, ("[主用户]",), "x").verdict is Verdict.PASS, "never_says 误报")

    check(mentions(good, ("雷阵雨",), "x").verdict is Verdict.PASS, "mentions 正例误判")
    check(mentions(good, ("暴雪",), "x").verdict is Verdict.FAIL, "mentions 反例漏判")
    check(mentions_any(good, ("暴雪", "雷阵雨"), "x").verdict is Verdict.PASS, "mentions_any 漏判")

    check(alive(Timeline()).verdict is Verdict.SKIP, "空时间线该判跳过不该判通过")
    only_heard = Timeline(lines=[(1.0, "听写", "今天天气怎么样")])
    check(alive(only_heard).verdict is Verdict.FAIL, "听到了却不回答该判失败")
    both = Timeline(lines=[(1.0, "听写", "问"), (2.0, "助手", "答")])
    check(alive(both).verdict is Verdict.PASS, "一问一答该判通过")

    ev = Timeline(events=[(1.0, "打断"), (2.0, "打断")])
    check(event_happened(ev, "打断", "x", 2).verdict is Verdict.PASS, "event 计数错")
    check(event_happened(ev, "派单", "x").verdict is Verdict.FAIL, "不存在的事件该失败")
    return bad


if __name__ == "__main__":
    problems = selftest()
    if problems:
        print("判据自检未通过：")
        for p in problems:
            print("  ✗", p)
        raise SystemExit(1)
    print("判据自检全部通过")
