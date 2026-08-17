"""跨会话的长期记忆：把该记住的事存在对话历史之外。

现在除了声纹档案，**什么都活不过重启**——上下文每次连接新建，任务登记表是进程内的
dict。上一轮对话里用户说过「我在北京」「以后回答简短点」，下一轮全忘光。

设计上跟这个项目已经踩明白的那条教训一致：**状态不能只存在于 LLM 的消息历史里**。
历史会被打断取消、被合并改写、被上下文长度挤掉——把长期事实放在那儿等于没放。所以
记忆是一份独立的存档，会话开始时**注入系统提示**，跟历史无关。

存什么：用户交代过的偏好和事实（住哪、怎么称呼、回答风格、纠正过的说法）。
**不存**对话流水——那是日志的事，塞进记忆只会让提示越来越长而没有新信息。

写入靠模型显式调 ``remember`` 工具。试过自动抽取，问题是没法判断一句话值不值得记，
结果要么记一堆废话，要么把听错的内容当事实记下来（中文听写在这个项目里错得不少）。
让模型显式判断，错了用户当场能纠正。

同一件事再说一遍会覆盖旧的（按 ``topic`` 去重），不会堆两条互相矛盾的。
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from loguru import logger

HERE = Path(__file__).parent
STORE = Path(os.getenv("MEMORY_PATH", HERE / "memory_store.jsonl"))
# 注入提示的条数上限。记忆是每轮都要重发的固定开销，太多会挤占上下文也拖慢首 token。
MAX_ITEMS = int(os.getenv("MEMORY_MAX", "40"))


@dataclass
class Fact:
    """记住的一件事。

    Parameters:
        topic: 归类用的短词（「住址」「称呼」「回答风格」）。同一个 topic 再写会覆盖，
            这样用户改主意时不会留下两条打架的记录。
        content: 事实本身，一句话。
        at: 记下的时刻。
    """

    topic: str
    content: str
    at: float = field(default_factory=time.time)


def load() -> list[Fact]:
    """读全部记忆，最近写的排在后面。"""
    if not STORE.exists():
        return []
    out: list[Fact] = []
    for line in STORE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(Fact(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue  # 存档坏了一行不该让整个记忆失效
    # 同一个 topic 只留最后一条。
    latest: dict[str, Fact] = {}
    for f in out:
        latest[f.topic] = f
    return sorted(latest.values(), key=lambda f: f.at)


def remember(topic: str, content: str) -> str:
    """记一件事，返回给模型看的确认语。"""
    topic, content = topic.strip()[:20], content.strip()[:120]
    if not topic or not content:
        return "没记住：内容是空的。"
    STORE.parent.mkdir(parents=True, exist_ok=True)
    with STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(Fact(topic, content)), ensure_ascii=False) + "\n")
    logger.info(f"记住了［{topic}］{content}")
    return f"记住了：{content}"


def forget(topic: str) -> str:
    """忘掉某个 topic 下的记忆。"""
    facts = [f for f in load() if f.topic != topic.strip()]
    _rewrite(facts)
    return f"忘掉了关于「{topic}」的记忆。"


def _rewrite(facts: list[Fact]) -> None:
    """整体重写存档。只在删除时用——正常写入一律追加。"""
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(
        "".join(json.dumps(asdict(f), ensure_ascii=False) + "\n" for f in facts),
        encoding="utf-8",
    )


def as_prompt() -> str:
    """把记忆拼成注入系统提示的一段。没有记忆就返回空串。"""
    facts = load()[-MAX_ITEMS:]
    if not facts:
        return ""
    lines = "；".join(f"{f.topic}：{f.content}" for f in facts)
    return (
        "\n以下是你之前记住的关于这个用户的事，直接当已知信息用，"
        "**不要主动复述、不要说「我记得你说过」这种话**，除非用户问起：\n" + lines
    )


def compact() -> None:
    """把存档压成去重后的样子。文件只增不减，跑久了值得清一次。"""
    facts = load()
    _rewrite(facts)
    logger.info(f"记忆已压缩到 {len(facts)} 条")


if __name__ == "__main__":
    for f in load():
        when = time.strftime("%m-%d %H:%M", time.localtime(f.at))
        print(f"  [{when}] {f.topic}：{f.content}")
    print(f"\n共 {len(load())} 条，存在 {STORE}")
