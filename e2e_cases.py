"""端到端场景与判据：每一条都对应一个**真实发生过**的故障。

不凭想象编场景。下面每条的 ``regression`` 写的是它当初怎么坏的——没有这一栏，
半年后没人知道这条为什么存在，也就没人敢删或敢改。

苛刻在三个地方：

* **打断**。用户在助手说话时插嘴，是这套系统最容易崩的地方，而随便聊天碰不到。
* **等待期间再问**。异步工作要五到十秒，用户等不了两秒就会重复问，一问就是打断。
* **不该回应时保持安静**。多人在场、旁人闲聊，助手接话就是错。

时间戳的用法：脚本里的秒数是**相对场景开始**的，判据也按这个时间轴写
（「第 12 秒之后必须再出声」），这样断言跟脚本对得上，改脚本时不会漏改判据。
"""

from dataclasses import dataclass, field

import checks
from checks import Timeline


@dataclass
class E2ECase:
    """一条端到端场景。

    Parameters:
        name: 场景名，命令行按它筛选。
        regression: 它当初是怎么坏的。**必填**——写不出来说明这条场景没有来由。
        script: ``(触发器, 用户说的话)``。触发器是数字时表示相对场景开始的秒数；
            是 ``"回应后+N"`` 时表示等助手开口之后再等 N 秒——**打断类场景必须用
            后者**，写死秒数会落在助手还没开口之前，测出来的东西跟想测的无关。
        total: 总时长，要留够后台任务跑完。
        checks: 判据函数，接收 ``Timeline`` 返回若干 ``Result``。
        flaky: 标记为易抖的场景。CI 里失败只警告不拦——但**必须写明理由**，
            不能拿它当万能挡箭牌。
    """

    name: str
    regression: str
    script: list[tuple[float, str]]
    total: float
    checks: object
    flaky: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def _interrupt_memory(tl: Timeline):
    """打断之后：不能失忆、不能复读。"""
    return [
        checks.no_repeat(tl, "打断后不复读"),
        checks.spoke_within(tl, 40.0, 30.0, "多轮打断后仍有响应"),
        checks.never_says(tl, ("[主用户]", "[说话人", "[skip]", "系统提示"), "内部标注没泄漏"),
    ]


def _async_delivery(tl: Timeline):
    """联网结果必须送达，哪怕中途被打断。"""
    return [
        checks.mentions_any(tl, ("我查一下", "查一下", "稍等"), "发起查询时先出声"),
        checks.spoke_within(tl, 25.0, 40.0, "打断后结果仍然送达"),
        checks.never_says(tl, ("[skip]", '"arguments"'), "没有泄漏协议片段"),
    ]


def _task_visible(tl: Timeline):
    """用户问「有什么在跑」时，搜索也要算在内。"""
    return [
        checks.never_says(tl, ("没有在跑的后台任务",), "不能说没有任务"),
        checks.spoke_within(tl, 12.0, 30.0, "查进度有响应"),
    ]


def _compute_dispatch(tl: Timeline):
    """算不出来不等于办不到，必须派给后台。"""
    return [
        checks.never_says(
            tl, ("没有这个能力", "无法计算", "我不能计算"), "不许直接说自己不会算"
        ),
        checks.spoke_within(tl, 3.0, 15.0, "有响应"),
    ]


def _no_fabrication(tl: Timeline):
    """陷阱题：不存在的东西不许编。"""
    return [
        checks.mentions_any(tl, ("没有", "不知道", "没查到", "不连"), "如实说没有"),
        checks.never_says(tl, ("PostgreSQL", "MySQL", "Redis", "MongoDB"), "没编造数据库"),
    ]


def _bystander(tl: Timeline):
    """旁人闲聊时助手该安静。"""
    return [
        checks.never_says(tl, ("[skip]", "[说话人"), "跳过标记没被念出来"),
    ]


CASES = [
    E2ECase(
        name="打断风暴",
        regression=(
            "助手被打断时 pipecat 默认把说了一半的回复整个丢掉，上下文里连着好几条"
            "用户发言没有助手消息，模型不记得说过什么就一直重复同一个答案。实测一段"
            "对话末尾连着四轮全是复读。修复见 filters.RememberInterrupted。"
        ),
        # 时间点跟着事件走：「打断」的定义是在助手说话时插进去，写死秒数会落在
        # 它还没开口之前（真实链路有七八秒往返）。
        script=[
            (0.5, "用两句话讲讲杭州这座城市"),
            ("回应后+1.5", "停，那苏州呢"),
            ("回应后+1.5", "算了，还是说杭州吧"),
            ("回应后+3", "你刚才讲杭州说了什么"),
        ],
        total=100.0,
        checks=_interrupt_memory,
        tags=("core", "interrupt"),
    ),
    E2ECase(
        name="查询期间打断",
        regression=(
            "联网查询要五到十秒，这期间助手一声不吭，用户等两秒就再问一遍，而再问就是"
            "一次打断——打断取消了「结果回来后念出来」的那轮 LLM，于是七次查询有四次"
            "查到了结果却一句没念。修复见 voice_bot._deliver：结果直接说出口，不依赖"
            "任何一轮 LLM 存活。"
        ),
        # 要在**查询已经发起、结果还没回来**的窗口里插话，所以等它说出
        # 「我查一下…」之后再开口。第一版写死第 4 秒，落在第一句还没听写出来之前，
        # 搜索压根没派出去，测的根本不是这件事。
        script=[
            (0.5, "帮我查一下后天上海的气温"),
            ("回应后+1", "你能听到我说话吗"),
        ],
        total=75.0,
        checks=_async_delivery,
        tags=("core", "async"),
    ),
    E2ECase(
        name="任务可见性",
        regression=(
            "联网查询从不登记进任务表，只有 agent 任务登记。用户问「现在有什么在跑」"
            "会被答「没有」，而当时其实有个搜索正在飞。修复见 progress_api 的 kind 字段。"
        ),
        # 同理：要等搜索真的发起了再问「有什么在跑」，否则答「没有」是对的。
        script=[
            (0.5, "帮我查一下北京下周有什么演唱会"),
            ("回应后+1", "你现在有什么任务在跑"),
        ],
        total=75.0,
        checks=_task_visible,
        tags=("core", "async"),
    ),
    E2ECase(
        name="算数要派后台",
        regression=(
            "让它算第 7856 个斐波那契数，它答「没有这个能力」。沙箱放行 python3，"
            "工具也在手边，卡点是提示把 agent 框死在「回答代码项目的问题」上。"
            "修复见 tasks.py 按任务类型换提示。"
        ),
        script=[(0.5, "帮我算一下第三百个斐波那契数是多少")],
        total=90.0,
        checks=_compute_dispatch,
        tags=("core", "agent"),
    ),
    E2ECase(
        name="不许编造",
        regression=(
            "问一个不存在的东西时模型倾向于编一个像样的答案，而不是说没有。"
            "实测被问「这个项目连的是哪个数据库」时答过「PostgreSQL、Redis」。"
        ),
        script=[(0.5, "这个项目连的是哪个数据库")],
        total=80.0,
        checks=_no_fabrication,
        tags=("core", "agent"),
    ),
    E2ECase(
        name="旁人闲聊",
        regression=(
            "多人在场时旁人的话会被听进来。助手要么接话（错），要么把内部的 [skip]"
            "标记念出来（更错——这个项目里内部标注被当成内容复述发生过三次）。"
        ),
        script=[
            (0.5, "今天天气怎么样"),
            ("回应后+1", "昨天那个会开到很晚才结束"),
        ],
        total=70.0,
        checks=_bystander,
        flaky="旁人这句是同一个音色合成的，声纹分不开，只能验「标记不泄漏」这一半",
        tags=("multi",),
    ),
]


def by_tag(tag: str) -> list[E2ECase]:
    return [c for c in CASES if tag in c.tags]


if __name__ == "__main__":
    for c in CASES:
        flag = "  [易抖]" if c.flaky else ""
        print(f"\n■ {c.name}（{c.total:.0f}s，标签 {'/'.join(c.tags)}）{flag}")
        print(f"  来由：{c.regression[:60]}…")
        for trigger, text in c.script:
            when = f"{trigger:5.1f}s" if isinstance(trigger, (int, float)) else f"{trigger:>7}"
            print(f"  [{when}] {text}")
    print(f"\n共 {len(CASES)} 条，总时长 {sum(c.total for c in CASES):.0f} 秒")
