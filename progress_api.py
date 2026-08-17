"""后台任务的进度中心：一份状态，三个出口。

同一份进度要喂三个地方——语音播报（"还在查，已经读了六个文件"）、用户主动问进度时
的口播，以及外部程序读的 HTTP 接口。所以状态集中放在这里，agent worker 负责写，
语音侧和 HTTP 路由负责读。

进程内共享，够用：agent worker 和语音管线跑在同一个 `WorkerRunner` 里。真要跨进程
（比如换成 Redis 总线把 agent 拆出去），把 ``REGISTRY`` 换成外部存储即可，读写接口
不用动。

HTTP 接口：

    GET /api/progress            # 全部任务
    GET /api/progress/{job_id}   # 单个任务
"""

import time
from dataclasses import asdict, dataclass, field

# job_id -> TaskProgress。任务结束后保留一小段时间，方便外部轮询拿到最终结果。
REGISTRY: dict[str, "TaskProgress"] = {}
KEEP_FINISHED = 300.0


@dataclass
class TaskProgress:
    """一个后台任务从派单到收尾的全部可见状态。

    Parameters:
        job_id: 任务 id。
        question: 派单时的问题。
        kind: ``agent`` 是后台 agent 任务，``search`` 是联网查询。两者都算「在跑的
            活儿」——用户问「有什么在跑」时指的是它俩，只登记一种就会答「没有」。
        state: ``running`` / ``paused`` / ``done`` / ``cancelled`` / ``error``。
        step: 进行到第几轮工具调用。
        tool_calls: 累计调用了多少次工具。
        files_read: 累计读了多少个文件。
        note: 当前正在做什么，人话。
        started_at: 派单时刻（epoch 秒）。
        running_at: 真正拿到并发名额、开始跑的时刻。和 ``started_at`` 的差就是
            排队等了多久。
        finished_at: 结束时刻，未结束为 None。
        answer: 最终答案，未结束为空。
    """

    job_id: str
    question: str
    kind: str = "agent"
    state: str = "running"
    step: int = 0
    tool_calls: int = 0
    files_read: int = 0
    note: str = "刚开始"
    started_at: float = field(default_factory=time.time)
    running_at: float | None = None
    finished_at: float | None = None
    answer: str = ""

    @property
    def elapsed(self) -> float:
        """从派单算起过了多少秒（含排队）。"""
        return (self.finished_at or time.time()) - self.started_at

    @property
    def queued(self) -> float:
        """在队列里等了多少秒。"""
        return (self.running_at - self.started_at) if self.running_at else 0.0

    @property
    def ran(self) -> float:
        """真正在跑的秒数，不含排队。"""
        if not self.running_at:
            return 0.0
        return (self.finished_at or time.time()) - self.running_at

    def as_dict(self) -> dict:
        """转成可 JSON 序列化的字典，附带算出来的耗时。"""
        data = asdict(self)
        data["elapsed_s"] = round(self.elapsed, 1)
        data["queued_s"] = round(self.queued, 1)
        data["ran_s"] = round(self.ran, 1)
        return data

    def spoken(self) -> str:
        """讲成一句适合朗读的中文进度。"""
        if self.state == "paused":
            return (
                f"那个任务先挂起了，已经跑了{int(self.elapsed)}秒、"
                f"读了{self.files_read}个文件，等前面的忙完接着跑。"
            )
        if self.state == "done":
            return f"那个任务已经查完了，一共跑了{int(self.elapsed)}秒。"
        if self.state == "cancelled":
            return "那个任务已经停了。"
        if self.state == "error":
            return "那个任务出错了。"
        parts = [f"跑了{int(self.elapsed)}秒", f"翻到第{self.step}轮"]
        if self.files_read:
            parts.append(f"读了{self.files_read}个文件")
        return f"{self.note}，{'，'.join(parts)}。"


def _prune():
    now = time.time()
    for job_id, task in list(REGISTRY.items()):
        if task.finished_at and now - task.finished_at > KEEP_FINISHED:
            REGISTRY.pop(job_id, None)


def start(job_id: str, question: str, kind: str = "agent") -> TaskProgress:
    """登记一个新任务。

    联网查询也要登记。它跟 agent 任务在用户眼里是一回事——都是「我让它去办、还没
    办完的事」——只有 agent 登记的话，用户问「现在有什么在跑」会得到「没有」，
    而当时其实有个搜索正在飞。
    """
    _prune()
    task = TaskProgress(job_id=job_id, question=question, kind=kind)
    REGISTRY[job_id] = task
    return task


def update(job_id: str, *, step: int, note: str, files_read: int = 0) -> None:
    """更新任务进度。``files_read`` 是本次新增的读文件数，累加。"""
    task = REGISTRY.get(job_id)
    if task is None:
        return
    task.step = step
    task.note = note
    task.tool_calls += 1
    task.files_read += files_read


def mark_started(job_id: str) -> None:
    """任务拿到并发名额、真正开跑。"""
    task = REGISTRY.get(job_id)
    if task and task.running_at is None:
        task.running_at = time.time()


def mark_paused(job_id: str, paused: bool) -> None:
    """标记任务挂起或恢复。挂起期间 elapsed 照走——用户关心的是等了多久。"""
    task = REGISTRY.get(job_id)
    if task is None:
        return
    task.state = "paused" if paused else "running"


def finish(job_id: str, *, state: str, answer: str = "") -> None:
    """给任务收尾。"""
    task = REGISTRY.get(job_id)
    if task is None:
        return
    task.state = state
    task.answer = answer
    task.finished_at = time.time()


def running() -> list[TaskProgress]:
    """当前还没结束的任务（含挂起的）。"""
    return [t for t in REGISTRY.values() if t.state in ("running", "paused")]


def snapshot() -> list[dict]:
    """全部任务的快照，新的在前。"""
    return [t.as_dict() for t in sorted(REGISTRY.values(), key=lambda t: -t.started_at)]


def install_routes(app) -> None:
    """把进度接口挂到 runner 的 FastAPI 实例上。

    Args:
        app: ``pipecat.runner.run.app``。
    """

    @app.get("/api/progress")
    async def all_progress():
        return {"tasks": snapshot(), "running": len(running())}

    @app.get("/api/progress/{job_id}")
    async def one_progress(job_id: str):
        task = REGISTRY.get(job_id)
        return task.as_dict() if task else {"error": "unknown job_id"}
