"""agent 的执行后端：自己写的工具循环，或者 Claude Agent SDK。

worker 那一层管的是调度、进度播报和取消，跟「谁来跑这个工具循环」无关。这里把循环
本身抽出来，两种实现共用一个接口：

* :class:`ToolLoopBackend` 走 OpenRouter 的 function calling，工具是 ``tools.py``
  里的 bash/browse/search。跟语音侧同一个 key、同一个模型，便宜且可控。
* :class:`ClaudeBackend` 把问题交给 Claude Agent SDK，用它自带的 Read/Grep/Glob/
  Bash/WebSearch。工具成熟得多，但要另一套鉴权，也另算钱。

两者都要往外吐两样东西，否则语音那边就成了黑箱：

* ``on_step(step, note)`` —— 每次工具调用报一句人话，用户问「查到哪了」时有话说。
* ``gate()`` —— 每轮开头 await 一下，调度器靠它实现挂起和恢复。

**挂起在两个后端里不是一回事。**自己写的循环停在两次 API 调用之间，停住就真的不再
花钱；Claude 那边跑在子进程里，停的只是我们这侧的读取，子进程该跑还是跑。所以对
``ClaudeBackend`` 而言 pause 只是「不再往下推进度」，省不下钱。
"""

import asyncio
import contextlib
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from loguru import logger

import tasks
import tools as T

# 重任务要读七八个文件。这条路上速度不是约束——用户听到的是「我去查一下」，几十秒
# 之后才要结果，多跑几轮换一个读全了的答案是划算的。
MAX_TURNS = int(os.getenv("AGENT_MAX_TURNS", "20"))

OnStep = Callable[[int, str, bool], Awaitable[None]]
Gate = Callable[[], Awaitable[None]]


class Usage:
    """一次运行烧掉的 token。

    两个后端在同一个模型上的差别几乎全在输入侧：SDK 每轮都要重发 Claude Code 的系统
    提示和全套工具 schema。没有这个数就只能拿墙钟时间比，而墙钟时间被网关排队掩盖了。
    """

    def __init__(self):
        self.input = 0
        self.cached = 0
        self.output = 0

    def reset(self) -> None:
        self.input = self.cached = self.output = 0

    def add(self, prompt: int, completion: int, cached: int = 0) -> None:
        """记一次调用。``cached`` 是命中缓存的输入，单独算——它的价钱只有正常输入的
        十分之一，混在一起会把 SDK 那边的开销夸大近十倍。"""
        self.input += prompt
        self.cached += cached
        self.output += completion

    def __str__(self) -> str:
        return f"输入 {self.input}（缓存 {self.cached}）/ 输出 {self.output} token"


SYSTEM = (
    "你是一个助手，用 shell、网页、搜索三样通用工具去弄清楚问题的答案。"
    "问本地代码就用 bash——ls 看结构、grep 找内容、cat 或 sed 读文件；"
    "**尽快把相关文件真正读一遍**，回答要基于文件内容，不要只看文件名猜。"
    "问外面的事（新闻、天气、行情、某个库的用法）用 search；"
    "拿到具体网址想看详情再用 browse。"
    "同一个意思不要反复换关键词搜，定位到了就去读。"
    "回答会被朗读出来，所以用口语化的中文，不要 markdown、不要代码块、不要列表符号，"
    "文件名和路径念出来即可。回答控制在三句话以内，抓住重点。"
)

GAVE_UP = "这个问题查得有点久，我还没找到确定的答案。"


class Backend(Protocol):
    """跑一轮 agentic 问答。"""

    async def run(
        self, question: str, *, on_step: OnStep, gate: Gate, kind: str = "general"
    ) -> str:
        """回答一个问题。

        Args:
            question: 用户的问题。
            on_step: 每次工具调用回调 ``(第几步, 一句人话, 这步是不是读了个文件)``。
            gate: 每轮开头 await，被挂起时会一直阻塞在这里。
            kind: 任务类型，见 ``tasks.py``。决定提示、工具集和轮数预算。

        Returns:
            给用户念的答案。
        """
        ...


def _describe(name: str, args: dict) -> str:
    """把一次工具调用讲成一句人话，用于进度播报。"""
    return {
        "bash": lambda: f"在跑 {args.get('command', '')[:40]}",
        "browse": lambda: f"在看 {args.get('url', '')[:40]}",
        "search": lambda: f"在搜 {args.get('query', '')[:30]}",
        "Bash": lambda: f"在跑 {args.get('command', '')[:40]}",
        "Read": lambda: f"在读 {Path(args.get('file_path', '')).name}",
        "Grep": lambda: f"在搜 {args.get('pattern', '')[:30]}",
        "Glob": lambda: f"在找 {args.get('pattern', '')[:30]}",
        "WebSearch": lambda: f"在搜 {args.get('query', '')[:30]}",
        "WebFetch": lambda: f"在看 {args.get('url', '')[:40]}",
    }.get(name, lambda: f"在跑 {name}")()


def _reads_file(name: str, args: dict) -> bool:
    """这次调用算不算「读了一个文件」，用于进度里的文件计数。"""
    if name == "Read":
        return True
    if name in ("bash", "Bash"):
        head = args.get("command", "").split()
        return bool(head) and Path(head[0]).name in ("cat", "head", "tail", "sed")
    return False


class ToolLoopBackend:
    """OpenAI 风格的 function calling 循环，工具来自 ``tools.py``。"""

    def __init__(self, client, model: str, root: Path):
        """初始化。

        Args:
            client: AsyncOpenAI 兼容客户端。
            model: 跑工具循环的模型。
            root: 项目根目录，bash 的工作目录锁在这里。
        """
        self._client = client
        self._model = model
        self._root = root
        self.usage = Usage()

    async def run(
        self, question: str, *, on_step: OnStep, gate: Gate, kind: str = "general"
    ) -> str:
        """跑工具循环直到模型不再调工具。"""
        self.usage.reset()
        task = tasks.pick(kind)
        # 只给这类任务用得上的工具。给多了会诱导它走弯路——算一道题时摆着搜索工具，
        # 它就可能去搜「第7856个斐波那契数」而不是自己算。
        allowed = [
            t for t in T.TOOLS if t["function"]["name"] in task.builtin_tools
        ] or T.TOOLS
        messages = [
            {"role": "system", "content": task.prompt.format(root=self._root)},
            {"role": "user", "content": question},
        ]
        for turn in range(task.max_turns):
            # 被挂起就停在这儿并让出名额，已经读过的文件和攒下的上下文都留着，
            # 恢复后接着跑。
            await gate()
            resp = await self._client.chat.completions.create(
                model=self._model, messages=messages, tools=allowed
            )
            if resp.usage:
                details = getattr(resp.usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                self.usage.add(
                    resp.usage.prompt_tokens - cached,
                    resp.usage.completion_tokens,
                    cached,
                )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return (msg.content or "").strip()

            messages.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                await on_step(turn + 1, _describe(name, args), _reads_file(name, args))
                result = await self._call_tool(name, args)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
        return GAVE_UP

    async def _call_tool(self, name: str, args: dict) -> str:
        """执行一次工具调用。

        任何异常都转成给模型看的文字，绝不往外抛——工具报错是它该自己处理的信息，
        炸掉循环等于整个任务白跑。
        """
        try:
            if name == "bash":
                # 工作目录锁在项目根，命令本身还要过 tools 里的只读白名单。
                return await T.run_bash(args["command"], self._root)
            if name == "browse":
                return await T.browse(args["url"])
            if name == "search":
                return await T.web_search(args["query"], self._client, self._model)
            return f"未知工具: {name}"
        except Exception as e:  # noqa: BLE001 - 工具报错要回给模型，不能炸掉循环
            return f"工具执行失败: {e}"


CLAUDE_SYSTEM = (
    # 项目路径必须写死在提示里。只设 `cwd` 不够——实测模型每一轮都会先去 Read
    # `~/.claude/projects/...` 下的同名文件（CLI 自己的会话目录），被路径检查拒掉
    # 之后再回来重找，每次白花两三轮，偶尔就此绕不出来把 max_turns 耗光。
    "你在回答关于一个代码项目的问题，项目在 {root}，所有文件都在这个目录下，"
    "不要去别的地方找。\n"
    "用 Read 读文件、Glob 找文件、Grep 搜内容、Bash 跑只读命令；"
    "问外面的事用 WebSearch，要看具体网址用 WebFetch。"
    "先把相关文件真正读一遍再回答，不要只看文件名猜。\n"
    # Glob 递归这件事必须讲明白。目录里常有 vendored 的第三方库源码，文件数以万计，
    # `Glob('*.py')` 会把它们全捞上来——实测模型据此答出「根目录下没有 python 文件」
    # 这种自相矛盾的话，而换成 find -maxdepth 1 每次都对。
    "注意 Glob 是**递归**的，会连子目录一起搜；只想看某一层（比如「根目录下有几个」）"
    "要用 Bash 跑 find 加 -maxdepth 1。子目录里可能有第三方库的完整源码，动辄上万个"
    "文件，除非问的就是那个库，否则别把它们算进来。\n"
    "回答会被朗读出来，所以用口语化的中文，不要 markdown、不要代码块、不要列表符号，"
    "文件名和路径念出来即可。回答控制在三句话以内。"
)

# Claude 侧只放行这些。没给 Edit/Write——语音很容易误触发，改文件的代价不可挽回。
CLAUDE_TOOLS = ["Read", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"]


def _gateway_env(model: str, base_url: str | None, token: str | None) -> dict:
    """指向第三方 Anthropic 兼容网关所需的环境变量。

    SDK 起的是 Claude Code 子进程，改地址只能靠环境变量。OpenRouter 的
    ``/api/v1/messages`` 就是 Anthropic 格式，GLM、DeepSeek 这些非 Anthropic 模型
    经它转译后一样能发 ``tool_use``。

    三个坑，少一个都跑不起来：

    * 地址要给到 ``/v1`` **之前**。CLI 自己会接 ``/v1/messages``，直接填 OpenRouter
      文档上那个带 ``/v1`` 的地址会变成 ``/api/v1/v1/messages``，报的却是「模型不存在」。
      所以这里主动把结尾的 ``/v1`` 削掉。
    * 小模型那两个变量得一起改。CLI 拿它跑标题生成之类的杂活，不改就会去请求网关
      不认识的 ``claude-haiku-*``。
    * 网关的模型名 CLI 不认识，得关掉上下文窗口校验，否则它按 200k 猜。

    Args:
        model: 网关上的模型全名。
        base_url: 网关地址，None 表示走官方。
        token: 网关的 key。

    Returns:
        传给子进程的环境变量，走官方时是空的。
    """
    if not base_url:
        return {}
    env = {
        "ANTHROPIC_BASE_URL": base_url.rstrip("/").removesuffix("/v1"),
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT": "1",
    }
    if token:
        env["ANTHROPIC_AUTH_TOKEN"] = token
    return env


class ClaudeBackend:
    """把问题交给 Claude Agent SDK 跑。

    每个问题起一个独立的 client：并发跑两个任务时共用一个会话会把上下文搅在一起，
    代价是每次多一次子进程启动，几十秒的任务摊得起。

    Bash 仍然过 ``tools._check_command``——SDK 自己的权限系统更完整，但这个项目对
    「语音误触发」的判断（只读白名单、路径不出项目目录）是一致的，两个后端共用一套
    策略比各写一套可靠。
    """

    def __init__(
        self,
        root: Path,
        model: str = "sonnet",
        max_turns: int = MAX_TURNS,
        base_url: str | None = None,
        auth_token: str | None = None,
    ):
        """初始化。

        Args:
            root: 允许探索的目录，也是 Bash 的工作目录。
            model: 传给 SDK 的模型别名或全名。指到第三方网关时要用那边的全名。
            max_turns: 单个问题最多几轮。
            base_url: Anthropic 兼容网关地址。给了就连它，不走官方。
            auth_token: 配套的 key。

        Raises:
            ImportError: 没装 ``claude-agent-sdk``。
        """
        try:
            from claude_agent_sdk import (
                ClaudeAgentOptions,
                ClaudeSDKClient,
                HookMatcher,
            )
        except ModuleNotFoundError as e:
            raise ImportError(
                "AGENT_BACKEND=claude 需要 `uv pip install claude-agent-sdk`"
            ) from e

        self._client_cls = ClaudeSDKClient
        self._options_cls = ClaudeAgentOptions
        self._hook_matcher = HookMatcher
        self._root = root
        self._model = model
        self._max_turns = max_turns
        self._env = _gateway_env(model, base_url, auth_token)
        self.usage = Usage()
        self._options = ClaudeAgentOptions(
            system_prompt=CLAUDE_SYSTEM.format(root=root),
            allowed_tools=CLAUDE_TOOLS,
            # 只读策略走 PreToolUse 钩子，不用 can_use_tool：``allowed_tools`` 里整个
            # 放行的工具会在回调之前就自动批准，那个回调根本不会被调用。
            #
            # 每个会碰路径的工具都要挂，只挡 Bash 不够——Read 接受绝对路径，
            # 光挡住 `cat /etc/passwd` 而放行 `Read(/etc/passwd)` 等于没挡。
            hooks={
                "PreToolUse": [
                    HookMatcher(matcher="Bash", hooks=[self._check_bash]),
                    *(
                        HookMatcher(matcher=name, hooks=[self._check_path])
                        for name in ("Read", "Glob", "Grep")
                    ),
                ]
            },
            cwd=str(root),
            model=model,
            max_turns=max_turns,
            env=_gateway_env(model, base_url, auth_token),
        )

    def _options_for(self, kind: str):
        """按任务类型造一份 options。

        三样东西跟着类型走：提示的第一句（决定它往哪使劲）、工具集（给不该用的工具
        只会诱导它走弯路）、轮数预算（算一道题两三轮够，翻代码要十几轮）。
        """
        task = tasks.pick(kind)
        hooks = {
            "PreToolUse": [
                self._hook_matcher(matcher="Bash", hooks=[self._check_bash]),
                *(
                    self._hook_matcher(matcher=name, hooks=[self._check_path])
                    for name in ("Read", "Glob", "Grep")
                ),
            ]
        }
        return self._options_cls(
            system_prompt=task.prompt.format(root=self._root),
            allowed_tools=task.tools,
            hooks=hooks,
            cwd=str(self._root),
            model=self._model,
            max_turns=task.max_turns,
            env=self._env,
        )

    async def _check_bash(self, data: dict, tool_use_id, context) -> dict:
        """每条 Bash 命令过一遍只读白名单，不合规就拒掉。"""
        command = (data.get("tool_input") or {}).get("command", "")
        return self._deny(T._check_command(command, self._root), command)

    async def _check_path(self, data: dict, tool_use_id, context) -> dict:
        """Read/Glob/Grep 的目标路径不能跑出项目目录。"""
        args = data.get("tool_input") or {}
        target = args.get("file_path") or args.get("path") or ""
        if not target or not target.startswith("/"):
            return {}  # 相对路径以 cwd 为基准，已经在项目里
        if T._inside(target, self._root):
            return {}
        return self._deny("路径不能跑出项目目录", target)

    def _deny(self, reason: str | None, what: str) -> dict:
        """把拒绝理由包成 PreToolUse 钩子的输出格式。"""
        if not reason:
            return {}
        logger.info(f"拒绝「{what}」：{reason}")
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    async def run(
        self, question: str, *, on_step: OnStep, gate: Gate, kind: str = "general"
    ) -> str:
        """把问题交给 SDK，边收边报进度。"""
        from claude_agent_sdk import AssistantMessage, ResultMessage

        client = self._client_cls(options=self._options_for(kind))
        await client.connect()
        answer, step = "", 0
        self.usage.reset()
        try:
            await client.query(prompt=question)
            async for msg in client.receive_response():
                # 挂起只是不再往下推进度：子进程那边照跑，省不下钱。
                await gate()
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        kind = type(block).__name__
                        if kind == "TextBlock":
                            answer += block.text
                        elif kind == "ToolUseBlock":
                            # 只留最后一次工具调用之后的文字。中间那些「我来看一下这个
                            # 文件」是说给自己听的旁白，念出来就成了车轱辘话。
                            answer = ""
                            step += 1
                            args = block.input or {}
                            await on_step(
                                step,
                                _describe(block.name, args),
                                _reads_file(block.name, args),
                            )
                elif isinstance(msg, ResultMessage):
                    u = msg.usage or {}
                    # 指到第三方网关时 total_cost_usd 是按 Anthropic 价目表算的，
                    # 对不上账，所以只记 token。
                    self.usage.add(
                        u.get("input_tokens", 0)
                        + u.get("cache_creation_input_tokens", 0),
                        u.get("output_tokens", 0),
                        u.get("cache_read_input_tokens", 0),
                    )
                    logger.info(
                        f"Claude 后端：{msg.num_turns} 轮，"
                        f"{msg.duration_ms / 1000:.1f} 秒，{self.usage}"
                    )
                    if msg.is_error and not answer:
                        if msg.subtype == "error_max_turns":
                            return GAVE_UP
                        return f"查的时候出错了：{msg.result or msg.subtype}"
        except asyncio.CancelledError:
            # 抢占：先让 SDK 停下再往外抛，否则子进程会留着跑完。中断本身失败无所谓，
            # 但不能盖掉真正的取消。
            with contextlib.suppress(Exception):
                await client.interrupt()
            raise
        finally:
            # 不用 `async with`：它的 __aexit__ 会吞掉外层的 CancelledError。
            await client.disconnect()

        return answer.strip() or GAVE_UP


GATEWAY = "https://openrouter.ai/api/v1"


def build(kind: str, *, client, model: str, root: Path) -> Backend:
    """按名字装一个后端，装不上就退回自带的循环。

    Claude 那条默认也走 OpenRouter（复用 ``OPENROUTER_API_KEY``），不额外要一套
    Anthropic 鉴权，也不另开一笔账。要连官方就把 ``CLAUDE_BASE_URL`` 设成空串。

    Args:
        kind: ``builtin`` 或 ``claude``。
        client: AsyncOpenAI 兼容客户端，自带循环用。
        model: 自带循环用的模型。
        root: 项目根目录。

    Returns:
        后端实例。
    """
    if kind == "claude":
        try:
            base_url = os.getenv("CLAUDE_BASE_URL", GATEWAY)
            backend = ClaudeBackend(
                root,
                # 效果优先选的（agent_model_bench.py，5 道跨文件难题）：
                # sonnet-4.5 答对 5/5、2.8 步、$0.095 每问；glm-4.6 4/5、3.4 步、$0.020；
                # haiku-4.5 3/5、3.6 步、$0.041（比 glm 又差又贵，别用）。
                # 差别不只在分数——问调度策略时 glm 答「队列缓冲、拒绝、降级」，是编的
                # 通用术语；sonnet 真去读了 scheduler.py，四个动作一字不差。
                # 这条路上慢一点贵一点都可以，答错不行。
                model=os.getenv("CLAUDE_AGENT_MODEL", "anthropic/claude-sonnet-4.5"),
                base_url=base_url,
                auth_token=os.getenv("CLAUDE_AUTH_TOKEN")
                or os.getenv("OPENROUTER_API_KEY"),
            )
            logger.info(f"agent 后端：Claude Agent SDK（{base_url or '官方'}）")
            return backend
        except ImportError as e:
            logger.warning(f"{e}，退回自带的工具循环")
    logger.info(f"agent 后端：自带工具循环（{model}）")
    return ToolLoopBackend(client, model, root)
