"""后台 agent worker：在语音对话之外跑多轮工具调用。

语音管线要在几百毫秒内响应，而 agentic 任务动辄十几秒——两者放同一条管线里必然
互相拖累。pipecat 的 worker 模型正好解决这件事：agent 作为独立 worker 挂在总线上，
语音那边通过 job RPC 派单，派完立刻继续说话，结果回来了再念出来。

这一层只管派单前后的事：改写听岔的提问、并发调度、进度播报、取消收尾。真正的工具
循环在 ``backends.py``，``AGENT_BACKEND`` 选用哪个。

工具只读。语音接口很容易被随口一句话触发，所以不给写文件的能力，路径也限制在
``root`` 之内。
"""

import asyncio
import os
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI
from pipecat.bus import BusJobRequestMessage, BusJobUpdateRequestMessage
from pipecat.pipeline.job_context import JobStatus
from pipecat.pipeline.job_decorator import job
from pipecat.workers.base_worker import BaseWorker

import backends
import progress_api
import scheduler as sched

# 派单前先让模型把听岔的提问改写一遍。设 REWRITE=0 关掉。
REWRITE = os.getenv("REWRITE", "1") != "0"
# 改写超时就用原文——它不值得让任务多等。
REWRITE_TIMEOUT = float(os.getenv("REWRITE_TIMEOUT", "6"))

REWRITE_SYSTEM = (
    "你在修一句被语音识别弄坏的中文提问。用户说的是关于一个代码项目的问题，"
    "识别时把专有名词听成了同音或近音的词——比如把「填充语」听成「春雨」、"
    "把「过滤器」听成「更集过滤器」、把英文文件名听成一串无意义的汉字。\n"
    "下面给你这个项目里真实存在的文件名。对照它们判断哪些词是听错的。\n"
    "判别标准很简单——看这个词在中文里成不成词：\n"
    "· 不成词的（「最急富败」「包士比亚」「更集」「商议及」「春雨」用在技术语境里），"
    "**一定是听错了，必须改**，念一念找出它对应哪个文件名或术语。\n"
    "· 正常的中文说法（「上一级目录」「整体架构」「延迟测试」），**一律不动**，"
    "哪怕它跟某个文件名有点像。把正常词硬套成文件名比不改还糟。\n"
    "其余部分一个字都不要动。只返回修好的那句话，不要解释、不要引号、不要思考过程。"
)


class ProjectAgentWorker(BaseWorker):
    """总线上的只读代码库问答 agent。

    没有 pipecat 管线，只从总线接 job 请求。并发上限由调度器管，超出的交给 LLM 决定
    是排队、抢占、挂起还是拒绝。
    """

    def __init__(
        self,
        name: str,
        *,
        root: Path,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        concurrency: int = 1,
        backend: str = "builtin",
    ):
        """初始化。

        Args:
            name: worker 名字，语音那边用它派单。
            root: 允许探索的项目根目录。工具不会走出这个目录。
            api_key: OpenRouter API key。
            model: 跑工具循环的模型，也用来做提问改写和调度决策。
            base_url: API 地址。
            concurrency: 同时真正在跑的任务数上限。超出的由调度器交给 LLM 决定
                是排队、抢占、挂起还是拒绝。
            backend: ``builtin`` 用自带的工具循环，``claude`` 交给 Claude Agent SDK。
        """
        super().__init__(name)
        self._root = root.resolve()
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._sched = sched.Scheduler(
            concurrency=concurrency, client=self._client, model=model
        )
        self._backend = backends.build(
            backend, client=self._client, model=model, root=self._root
        )

    @job(name="ask")
    async def on_ask(self, message: BusJobRequestMessage) -> None:
        """回答一个关于项目的问题，过程中持续上报进度。"""
        question = message.payload.get("question", "")
        kind = message.payload.get("kind", "general")
        logger.info(f"Worker '{self.name}': 收到问题「{question}」")
        progress_api.start(message.job_id, question)
        handle = self._sched.register(message.job_id, question)
        try:
            if not await self._admit(handle):
                progress_api.finish(message.job_id, state="cancelled")
                await self.send_job_response(
                    message.job_id, {"answer": "这个先不查了。"}
                )
                return
            answer = await self._run_agent(question, message.job_id, handle, kind)
            logger.info(f"Worker '{self.name}': 回答完成（{len(answer)} 字）")
            progress_api.finish(message.job_id, state="done", answer=answer)
            await self.send_job_response(message.job_id, {"answer": answer})
        except asyncio.CancelledError:
            logger.info(f"Worker '{self.name}': 任务被取消")
            progress_api.finish(message.job_id, state="cancelled")
            raise
        except Exception as e:
            logger.error(f"Worker '{self.name}': 出错 {e}")
            progress_api.finish(message.job_id, state="error", answer=str(e))
            await self.send_job_response(
                message.job_id, {"error": str(e)}, status=JobStatus.ERROR
            )
        finally:
            self._sched.release(handle)
            self._sched.forget(message.job_id)
            # 名额空出来了，把挂起最久的那个放回来。
            self._sched.resume_one()

    async def _admit(self, handle: sched.TaskHandle) -> bool:
        """让新任务进场。

        名额没满就直接占一个。满了先问 LLM 怎么安排，再按决策执行——抢占和挂起都
        会腾出名额，所以之后照常等 ``acquire``。

        Returns:
            True 表示可以开跑，False 表示这个任务被判为不必做。
        """
        if len(self._sched.running()) < self._sched.limit:
            await self._sched.acquire(handle)
            return True

        decision = await self._sched.decide(handle.question)
        logger.info(
            f"调度决策 {decision.action}"
            f"{' -> ' + decision.target[:8] if decision.target else ''}"
            f"：{decision.reason}"
        )
        # 让用户知道后台被怎么安排了——调度是它看不见的，不说一声就成了黑箱。
        await self._say(handle.job_id, decision.say)
        if decision.action == sched.REJECT:
            return False
        if decision.action == sched.PREEMPT and decision.target:
            await self.cancel_running(decision.target)
        elif decision.action == sched.PAUSE and decision.target:
            self._sched.pause(decision.target)
        # queue 什么都不用做：下面这行会一直等到有名额。
        await self._sched.acquire(handle)
        return True

    async def _say(self, job_id: str, text: str) -> None:
        """把一句话以 urgent 更新推给语音侧，让它立刻念出来。"""
        if not text:
            return
        try:
            await self.send_job_update(
                job_id, {"spoken": text, "urgent": True, "note": text}, urgent=True
            )
        except RuntimeError:
            pass

    async def cancel_running(self, job_id: str) -> None:
        """抢占：把某个在跑的任务取消掉。"""
        task = self._job_handler_tasks.get(job_id)
        if task:
            task.cancel()

    async def on_job_update_requested(
        self, message: BusJobUpdateRequestMessage
    ) -> None:
        """用户中途问「查得怎么样了」时，把当前进度立刻回过去。

        用 ``urgent=True`` 发：这是用户在等的回答，不该排在积压的普通消息后面。
        """
        task = progress_api.REGISTRY.get(message.job_id)
        payload = task.as_dict() if task else {"step": 0, "note": "刚开始"}
        payload["spoken"] = task.spoken() if task else "刚开始查。"
        payload["urgent"] = True
        await self.send_job_update(message.job_id, payload, urgent=True)

    async def _report(self, job_id: str | None, step: int, note: str, *, files: int = 0) -> None:
        """把一步进度写进 registry，并推给语音侧。"""
        if job_id is None:
            return
        progress_api.update(job_id, step=step, note=note, files_read=files)
        task = progress_api.REGISTRY.get(job_id)
        payload = task.as_dict() if task else {"step": step, "note": note}
        payload["spoken"] = task.spoken() if task else note
        try:
            await self.send_job_update(job_id, payload)
        except RuntimeError:
            # 任务刚被取消，job 已经从活跃表里摘掉了。进度报不出去无所谓，
            # 但不能让它把循环炸掉——外层还要走取消收尾。
            pass

    async def _rewrite(self, question: str) -> str:
        """把听岔的提问改写回来。

        `TermCorrection` 那张硬编码词表只挡得住反复出现的固定错法，遇到「填充语」被
        听成「春雨」这种就没辙。这里换成让模型对着项目里的真实文件名去猜——它比词表
        通用，而且这一步跑在 agent 侧，加的一秒多摊在几十秒的任务里可以忽略，不会
        动到语音那条链路的延迟。

        Args:
            question: 听写出来的原始提问。

        Returns:
            改写后的提问。任何失败都原样返回——改写是锦上添花，不该挡住任务。
        """
        try:
            names = sorted(
                p.name
                for p in self._root.glob("*.py")
                if p.is_file()
            )[:60]
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": REWRITE_SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"项目文件：{'、'.join(names)}\n\n提问：{question}"
                            ),
                        },
                    ],
                    # 输出就一句话。不封顶的话模型会写一大段推理，实测能跑到 90 秒。
                    max_tokens=120,
                    temperature=0,
                ),
                timeout=REWRITE_TIMEOUT,
            )
            fixed = (resp.choices[0].message.content or "").strip()
        except (Exception, asyncio.TimeoutError) as e:  # noqa: BLE001
            # 改写是锦上添花，超时或出错一律用原文，绝不阻塞任务。
            logger.warning(f"提问改写失败，用原文：{e}")
            return question

        # 模型可能话痨或者跑偏，长度差太多就不信它。
        if not fixed or len(fixed) > len(question) * 2 + 20:
            return question
        if fixed != question:
            logger.info(f"提问改写：{question!r} -> {fixed!r}")
        return fixed

    async def _run_agent(
        self,
        question: str,
        job_id: str | None = None,
        handle: sched.TaskHandle | None = None,
        kind: str = "general",
    ) -> str:
        """改写提问，然后交给后端跑，中间把进度和调度闸门接上。"""
        if REWRITE:
            question = await self._rewrite(question)

        async def on_step(step: int, note: str, reads_file: bool) -> None:
            await self._report(job_id, step, note, files=int(reads_file))

        async def gate() -> None:
            if handle:
                await self._sched.gate(handle)

        return await self._backend.run(
            question, on_step=on_step, gate=gate, kind=kind
        )


def build_agent_worker(name: str = "project-agent") -> ProjectAgentWorker:
    """按环境变量装配 worker。

    Args:
        name: worker 名字。

    Returns:
        配置好的 worker。``PROJECT_PATH`` 指定探索目录（默认当前目录），
        ``AGENT_MODEL`` 指定跑工具循环的模型，``AGENT_CONCURRENCY`` 指定同时
        真正在跑几个任务（默认 2），``AGENT_BACKEND`` 选 ``builtin`` 还是
        ``claude``。
    """
    return ProjectAgentWorker(
        name,
        root=Path(os.getenv("PROJECT_PATH", ".")),
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.getenv("AGENT_MODEL", "deepseek/deepseek-v4-flash"),
        # 每个在跑的任务在 claude 后端下是一个独立的 CLI 子进程，抢的是本机 CPU，
        # 而听写和合成也在同一批核上——并发开太大先垮的是语音那条路，不是 agent。
        # 4 是留了余量的默认值，真要往上推先用 perf_probe.py 量一遍。
        concurrency=int(os.getenv("AGENT_CONCURRENCY", "4")),
        # 默认走 Claude Agent SDK：难题上它明显更肯去读代码，自带循环有时凭常识
        # 硬答（`backend_bench.py` 的五道跨文件题，10 对 8）。这条路上慢一点无所谓。
        backend=os.getenv("AGENT_BACKEND", "claude"),
    )
