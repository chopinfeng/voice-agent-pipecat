"""DeepSeek Harness 后端：把 agentic 任务交给 ``dsh`` 的 headless 模式跑。

跟 ``ClaudeBackend`` 并列的第三个后端，共用 ``backends.Backend`` 那个接口。它是
Node 写的，只能起子进程——不像 Claude Agent SDK 有 Python 绑定。

**两个能力上的硬差距，先说清楚免得对比时得出错误结论：**

* **没有分步事件。**``dsh --profile headless "任务"`` 只在结束时打印最终答案，中间
  一个字都不吐。所以 ``on_step`` 回调基本是空的——用户问「查到哪一步了」时，这个
  后端答不上来，而 Claude SDK 那边能报「在读 filler.py」。
* **没有工具/轮数开关。**任务类型（``tasks.py``）在这里只能通过**改写提问**来体现，
  没法像 Claude 侧那样按类型换工具集和轮数预算。

配置走 ``$DSH_HOME``：``settings.yaml`` 定义 provider，``cordis.patch.yml`` 定默认模型，
运行时生成。有 ``DEEPSEEK_API_KEY`` 就走**官方 provider**（``deepseek-official``），
那是这套 harness 的原生路径；没有则退回 OpenRouter 网关，代价是模型名要写成
``deepseek/deepseek-v4-flash`` 这种带前缀的形式。

权限用 ``DSH_PERMISSION_MODE=read-only``。注意它的审批策略是**除了
``danger-full-access`` 都要 ask**，headless 下没人可问，所以只读模式下模型碰到写操作
会卡住而不是被拒——任务提示里要写明只读。
"""

import asyncio
import os
import shutil
from pathlib import Path

from loguru import logger

import tasks

HERE = Path(__file__).parent
DSH_HOME = Path(os.getenv("DSH_HOME", HERE / "testdata" / "dshhome"))
# 一个任务最多跑这么久。dsh 自己没有轮数上限可配，只能从外面掐时间。
TIMEOUT = float(os.getenv("DSH_TIMEOUT", "300"))


def _write_config(model: str, base_url: str) -> None:
    """生成 dsh 的配置。

    有官方 key 就用官方 provider——它是 harness 自带的，模型名用裸名
    （``deepseek-v4-flash``）。没有才退回自定义 provider 指向网关，那时模型名要带
    厂商前缀。每次运行都重写，免得改了模型却用着上次的配置。
    """
    DSH_HOME.mkdir(parents=True, exist_ok=True)
    official = bool(os.getenv("DEEPSEEK_API_KEY"))
    if official:
        # 官方 provider 已在 harness 的 catalog 里，只要有 key 就能用，不必自定义。
        (DSH_HOME / "settings.yaml").write_text("{}\n", encoding="utf-8")
        provider = "deepseek-official"
    else:
        (DSH_HOME / "settings.yaml").write_text(
            "llm-pi-ai:\n"
            "  providers:\n"
            "    openrouter:\n"
            "      name: OpenRouter\n"
            "      apiKeyEnv: OPENROUTER_API_KEY\n"
            "      api: openai-completions\n"
            f"      baseURL: {base_url}\n"
            "      models:\n"
            f"        - id: {model}\n",
            encoding="utf-8",
        )
        provider = "openrouter"
    (DSH_HOME / "cordis.patch.yml").write_text(
        "- id: agent-default-model\n"
        "  config:\n"
        f"    provider: {provider}\n"
        f"    model: {model}\n",
        encoding="utf-8",
    )


class DshBackend:
    """用 DeepSeek Harness 的 headless 模式跑一个任务。"""

    def __init__(
        self,
        root: Path,
        model: str = "deepseek-v4-flash",
        base_url: str = "https://openrouter.ai/api/v1",
    ):
        """初始化。

        Args:
            root: agent 的工作目录。dsh 用**调用进程的 cwd** 当工作区，没有单独的
                参数，所以这里靠子进程的 cwd 传进去。
            model: OpenRouter 上的模型全名。
            base_url: 网关地址。

        Raises:
            RuntimeError: 找不到 npx。
        """
        if not shutil.which("npx"):
            raise RuntimeError("DeepSeek Harness 要 Node/npx，没装")
        self._root = root
        self._model = model
        self._base_url = base_url
        self.usage = None  # dsh 不吐 token 用量，没法跟别的后端比成本
        _write_config(model, base_url)

    async def run(
        self, question: str, *, on_step, gate, kind: str = "general"
    ) -> str:
        """跑一个任务，返回最终答案。

        ``on_step`` 只在开头报一次「已经交给 dsh」——headless 模式中途不吐任何事件，
        报不出真实进度。这是这个后端相对 Claude SDK 的实打实的功能缺失。
        """
        await gate()
        task = tasks.pick(kind)
        # 类型只能通过改写提问体现：dsh 没有按任务换工具集和轮数的开关。
        prompt = f"{task.prompt.format(root=self._root)}\n\n任务：{question}"
        await on_step(1, f"交给 DeepSeek Harness（{kind}）", False)

        env = {
            **os.environ,
            "DSH_HOME": str(DSH_HOME),
            # 只读。注意 dsh 的审批策略在非 danger 模式下是 ask，headless 没人可问，
            # 所以提示里也写了只读，双保险。
            "DSH_PERMISSION_MODE": "read-only",
            "DSH_TELEMETRY_MODE": "DISABLED",
        }
        proc = await asyncio.create_subprocess_exec(
            "npx",
            "-y",
            "@deepseek-ai/dsh",
            "--profile",
            "headless",
            prompt,
            cwd=str(self._root),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=TIMEOUT)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning(f"dsh 超过 {TIMEOUT:.0f} 秒未结束，已中止")
            return "这个问题查得有点久，我还没找到确定的答案。"

        answer = out.decode("utf-8", errors="replace").strip()
        if not answer:
            tail = err.decode("utf-8", errors="replace").strip()[-200:]
            logger.warning(f"dsh 没有输出：{tail}")
            return f"查的时候出错了：{tail[:80]}"
        return answer
