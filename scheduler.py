"""后台任务的并发闸门和调度决策。

并发上限是配置项（``AGENT_CONCURRENCY``），但**满了之后怎么办**不是——那是个需要
看具体内容的判断：用户刚提的这个急不急？在跑的那几个里有没有已经跑了两分钟、
再等等就出结果的？有没有哪个其实已经没意义了（用户改了主意、问的是同一件事）？
这类取舍写死成规则只会僵，所以交给 LLM：把在跑的任务状态和新任务一起给它，
让它选排队、抢占、暂停还是拒绝。

暂停是逐轮生效的。agent 的工具循环每跑完一轮就看一眼闸门，暂停时把并发名额让出来，
恢复时重新排队获取——所以暂停不会丢掉已经读过的文件和已经攒下的上下文。
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

from loguru import logger

import progress_api

# LLM 只能返回这几种决策。
QUEUE = "queue"  # 排队等，前面的跑完再说
PREEMPT = "preempt"  # 干掉某个在跑的，让新的立刻上
PAUSE = "pause"  # 挂起某个在跑的，让新的先上，之后再恢复
REJECT = "reject"  # 不接这个新任务

# target 用列表里的序号而不是 job_id：让模型复述一个 UUID 很容易漏或写错，实测它
# 干脆不填。序号必填（用不上时填 0），schema 强制它给个值。
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [QUEUE, PREEMPT, PAUSE, REJECT]},
        "target": {
            "type": "integer",
            "description": (
                f"{PREEMPT} 和 {PAUSE} 时填要动的那个任务的序号（从 1 开始），"
                f"{QUEUE} 和 {REJECT} 时填 0。"
            ),
        },
        "reason": {
            "type": "string",
            "description": "你这么判断的依据，只进日志，用户看不到，可以写详细。",
        },
        "say": {
            "type": "string",
            "description": (
                "念给用户听的一句大白话，二十字以内，比如「先把手头这个跑完再查你说的」。"
                "只说结论、不说推理过程，不要出现任务序号、文件名、秒数这些内部细节。"
            ),
        },
    },
    "required": ["action", "target", "reason", "say"],
}

SYSTEM = (
    "你是语音助手的后台任务调度员。并发名额已经用满，现在来了一个新任务，"
    "你要决定怎么安排。可选：\n"
    f"- {QUEUE}：让新任务排队。在跑的都还有价值、而且快出结果时选这个。\n"
    f"- {PREEMPT}：取消掉某个在跑的任务，让新任务立刻上。只有当那个任务明显没意义了"
    "（用户已经改了主意、或者和新任务问的是同一件事）才选。\n"
    f"- {PAUSE}：挂起某个在跑的，让新任务先跑，之后再恢复它。新任务明显更急、"
    "而被挂起的那个跑了很久还没完时选这个——它的进度会保留。\n"
    f"- {REJECT}：不接新任务。只有新任务本身没意义时才选。\n"
    "优先考虑用户最近的意图。跑了很久眼看要出结果的任务，不要轻易取消。"
)


@dataclass
class TaskHandle:
    """一个后台任务的闸门。

    Parameters:
        job_id: 任务 id。
        question: 任务问题。
        resume: 置位表示可以跑，清位表示暂停在下一轮开头。
        holds_slot: 当前是否占着并发名额。
        first_slot_at: 第一次拿到名额的时刻。在此之前都是在排队，把两段分开才
            看得出并发到底是缩短了执行还是只是缩短了等待。
    """

    job_id: str
    question: str
    resume: asyncio.Event = field(default_factory=asyncio.Event)
    holds_slot: bool = False
    first_slot_at: float | None = None

    def __post_init__(self):
        self.resume.set()


@dataclass
class Decision:
    """调度结果。"""

    action: str
    target: str | None = None
    reason: str = ""
    say: str = ""


class Scheduler:
    """并发闸门 + LLM 调度决策。"""

    def __init__(self, *, concurrency: int, client, model: str):
        """初始化。

        Args:
            concurrency: 同时真正在跑的任务数上限。
            client: OpenAI 兼容客户端，用来做调度判断。
            model: 判断用的模型。
        """
        self._limit = max(1, concurrency)
        self._sem = asyncio.Semaphore(self._limit)
        self._client = client
        self._model = model
        self._tasks: dict[str, TaskHandle] = {}

    @property
    def limit(self) -> int:
        """并发上限。"""
        return self._limit

    def running(self) -> list[TaskHandle]:
        """当前占着名额的任务。"""
        return [t for t in self._tasks.values() if t.holds_slot]

    def paused(self) -> list[TaskHandle]:
        """被挂起、等着恢复的任务。"""
        return [t for t in self._tasks.values() if not t.resume.is_set()]

    def register(self, job_id: str, question: str) -> TaskHandle:
        """登记一个任务，返回它的闸门。"""
        handle = TaskHandle(job_id=job_id, question=question)
        self._tasks[job_id] = handle
        return handle

    def forget(self, job_id: str) -> None:
        """任务结束，注销。"""
        self._tasks.pop(job_id, None)

    async def decide(self, new_question: str) -> Decision:
        """名额满了，问 LLM 怎么安排。

        Args:
            new_question: 新来的任务问题。

        Returns:
            决策。LLM 出错或返回不可用内容时退回排队——排队最保守，不会丢任务。
        """
        running = self.running()
        if not running:
            return Decision(action=QUEUE, reason="没有在跑的任务", say="这就去查")

        snapshot = []
        for index, handle in enumerate(running, 1):
            task = progress_api.REGISTRY.get(handle.job_id)
            snapshot.append(
                {
                    "序号": index,
                    "问题": handle.question,
                    "已跑秒数": round(task.elapsed, 1) if task else 0,
                    "读了几个文件": task.files_read if task else 0,
                    "当前在做": task.note if task else "",
                }
            )

        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"并发上限 {self._limit}，已满。\n"
                            f"在跑的任务：{json.dumps(snapshot, ensure_ascii=False)}\n"
                            f"新来的任务：{new_question}"
                        ),
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "decision", "schema": DECISION_SCHEMA},
                },
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001 - 调度失败不该把任务弄丢
            logger.warning(f"调度决策失败，退回排队：{e}")
            return Decision(action=QUEUE, reason="调度判断失败，先排队", say="排上了，稍等")

        action = data.get("action", QUEUE)
        index = data.get("target") or 0
        target = None
        if action in (PREEMPT, PAUSE):
            if not 1 <= index <= len(running):
                logger.warning(f"调度给的序号 {index} 越界，退回排队")
                return Decision(action=QUEUE, reason="排队等前面的跑完", say="排在前面那个后面")
            target = running[index - 1].job_id
        # 模型偶尔会把整段推理塞进给用户的那句里，兜一道底：太长就截，
        # 空了就用一句通用的。
        say = (data.get("say") or "").strip()
        if not say or len(say) > 40:
            say = {PREEMPT: "先停下手头那个，优先给你查这个",
                   PAUSE: "先把手头那个挂起，回头接着跑",
                   REJECT: "这个先不查了",
                   }.get(action, "排上了，稍等")
        return Decision(
            action=action, target=target, reason=data.get("reason", ""), say=say
        )

    async def acquire(self, handle: TaskHandle) -> None:
        """占一个并发名额，占不到就等。"""
        await self._sem.acquire()
        handle.holds_slot = True
        if handle.first_slot_at is None:
            handle.first_slot_at = time.monotonic()
            progress_api.mark_started(handle.job_id)

    def release(self, handle: TaskHandle) -> None:
        """让出并发名额。"""
        if handle.holds_slot:
            handle.holds_slot = False
            self._sem.release()

    def pause(self, job_id: str) -> bool:
        """挂起一个任务。它会在下一轮开头停住并让出名额。"""
        handle = self._tasks.get(job_id)
        if not handle or not handle.resume.is_set():
            return False
        handle.resume.clear()
        logger.info(f"调度：挂起「{handle.question[:20]}」")
        return True

    def unpause(self, job_id: str) -> bool:
        """恢复一个被挂起的任务。"""
        handle = self._tasks.get(job_id)
        if not handle or handle.resume.is_set():
            return False
        handle.resume.set()
        logger.info(f"调度：恢复「{handle.question[:20]}」")
        return True

    def resume_one(self) -> str | None:
        """名额空出来时，挑一个挂起的任务恢复。

        Returns:
            被恢复的 job_id，没有可恢复的返回 None。
        """
        waiting = self.paused()
        if not waiting:
            return None
        # 先挂起的先恢复，避免某个任务被反复插队饿死。
        oldest = min(
            waiting,
            key=lambda h: (
                progress_api.REGISTRY[h.job_id].started_at
                if h.job_id in progress_api.REGISTRY
                else 0
            ),
        )
        self.unpause(oldest.job_id)
        return oldest.job_id

    async def gate(self, handle: TaskHandle) -> None:
        """每轮工具循环开头过一次闸门。

        暂停期间把名额让出去，恢复时重新排队获取——这样挂起的任务不会占着名额，
        但已经读过的文件和攒下的上下文都还在。
        """
        if handle.resume.is_set():
            return
        self.release(handle)
        progress_api.mark_paused(handle.job_id, True)
        await handle.resume.wait()
        progress_api.mark_paused(handle.job_id, False)
        await self.acquire(handle)
