"""给 agent 的通用工具：shell、抓网页、联网搜索。

不做「查天气」「查股价」这种按场景定制的工具——那样每来一个新需求就得加一个函数，
而且模型只能在你预设过的路子里打转。给通用能力，让它自己组合：查文件用 shell 的
``ls``/``grep``，看网页用 ``browse``，不知道的事用 ``search``。

**shell 默认只读。**语音接口特别容易误触发——听写把「上一级目录」听成「商议及目录」
这种事在这个项目里反复出现过，要是这时候执行的是 ``rm``，代价不可挽回。所以默认
只放行一批只读命令，写操作要显式开 ``ALLOW_WRITE=1``，而且无论开不开都挡住
``sudo``、管道重定向到文件、以及删除整棵目录树这类。
"""

import asyncio
import os
import re
import shlex
from pathlib import Path

import httpx
from loguru import logger

# 只读命令白名单。够用来翻代码、看结构、查历史。
READONLY_CMDS = {
    "ls", "cat", "head", "tail", "wc", "grep", "rg", "find", "file", "stat",
    "tree", "du", "df", "date", "pwd", "echo", "sort", "uniq", "cut", "awk",
    "sed", "diff", "which", "basename", "dirname", "realpath", "git", "jq",
    "python3", "python",
}
# git 里也有会改东西的子命令，单独挡一下。
GIT_WRITE = {"commit", "push", "reset", "rebase", "checkout", "merge", "clean", "rm"}
# 无论如何都不放行的。重定向那两条放行 /dev/null——`2>/dev/null` 是模型写命令时
# 的默认习惯，一律拒掉的话它会反复换写法试探，白白多跑好几轮。
ALWAYS_BLOCKED = re.compile(
    r"\bsudo\b|\bsu\b|\bchmod\b|\bchown\b|\bkill\b|\bshutdown\b|\breboot\b|"
    r"\bmkfs\b|\bdd\b|\bcurl\b|\bwget\b|\bnc\b|\bssh\b|\bscp\b|"
    r"rm\s+-[rf]|>>?\s*/(?!dev/null)"
)

ALLOW_WRITE = os.getenv("ALLOW_WRITE") == "1"
BASH_TIMEOUT = float(os.getenv("BASH_TIMEOUT", "20"))
MAX_OUTPUT = 8000

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "在项目目录下执行 shell 命令，看输出。查文件用 ls/find，"
                "搜内容用 grep，读文件用 cat/head/sed。默认只读，写操作会被拒。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse",
            "description": "抓一个网页，返回正文文本。需要看某个具体网址的内容时用。",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "联网搜索，查你不知道的事：新闻、天气、股价、汇率、某个库的最新用法。"
                "返回带来源的简要回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _inside(token: str, root: Path) -> bool:
    """这个绝对路径是不是落在 root 里面。

    不用 ``resolve()``：那会跟着符号链接走，项目里一个指向外面的链接就能绕过检查。
    """
    try:
        return Path(os.path.normpath(token)).is_relative_to(root)
    except (ValueError, OSError):
        return False


def _check_command(command: str, root: Path | None = None) -> str | None:
    """检查命令能不能跑，不能就返回拒绝理由。

    Args:
        command: 完整命令行。
        root: 给了就允许指向它内部的绝对路径。不给则一概拒绝绝对路径——够用，
            但会跟习惯写绝对路径的调用方打架。

    Returns:
        None 表示放行，否则是拒绝理由。
    """
    if ALWAYS_BLOCKED.search(command):
        return "这条命令里有不允许的操作（提权、网络下载、递归删除或写系统路径）"

    try:
        parts = shlex.split(command)
    except ValueError as e:
        return f"命令解析不了：{e}"
    if not parts:
        return "空命令"

    # 管道和 && 串起来的每一段都要查，不能只看第一个词。
    segments, current = [], []
    for token in parts:
        if token in ("|", "&&", ";", "||"):
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)

    # 路径不能跑出工作目录。换成 shell 之后原来那套 `_safe_path` 校验就没了——
    # 实测 `cat ../../../etc/passwd` 会真的执行（只是那个路径碰巧不存在）。
    # shell 里的路径没法静态分析全（`cat $(ls ..)` 这种），所以直接拦掉字面量：
    # 参数里出现 `..` 或者以 `/` 开头，一律不放行。
    for token in parts:
        if token.startswith("-"):
            continue  # 选项不是路径，`sed -n 1,10p` 这种别误伤
        if token == ".." or token.startswith("../") or "/../" in token:
            return "路径不能跑出项目目录"
        if token.startswith("/") and token != "/dev/null":
            if root and _inside(token, root):
                continue
            return "路径不能跑出项目目录"

    for seg in segments:
        if not seg:
            continue
        cmd = Path(seg[0]).name
        if ALLOW_WRITE:
            continue
        if cmd not in READONLY_CMDS:
            return f"「{cmd}」不在只读白名单里。要跑写操作得开 ALLOW_WRITE=1"
        if cmd == "git" and len(seg) > 1 and seg[1] in GIT_WRITE:
            return f"git {seg[1]} 会改动仓库，只读模式下不放行"
    return None


async def run_bash(command: str, cwd: Path) -> str:
    """在指定目录下跑一条 shell 命令。

    Args:
        command: 命令行。
        cwd: 工作目录，命令只在这里跑。

    Returns:
        命令输出，或者拒绝/出错的说明。
    """
    reason = _check_command(command, cwd)
    if reason:
        logger.info(f"拒绝执行「{command}」：{reason}")
        return f"没执行：{reason}"

    logger.debug(f"执行：{command}")
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=BASH_TIMEOUT)
    except asyncio.TimeoutError:
        proc.kill()
        return f"命令跑了超过 {BASH_TIMEOUT:.0f} 秒还没结束，已中止"
    except Exception as e:  # noqa: BLE001 - 执行失败要回给模型，不能炸掉循环
        return f"执行失败：{e}"

    text = out.decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n…（输出太长，截断在 {MAX_OUTPUT} 字符）"
    return text or "（没有输出）"


_TAG = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML = re.compile(r"<[^>]+>")


async def browse(url: str, timeout: float = 15.0) -> str:
    """抓一个网页，粗略转成正文。

    没上 readability 那类库——先看看够不够用。脚本样式先剥掉再去标签，否则会把
    一堆 JS 当正文返回。

    Args:
        url: 网址。
        timeout: 超时秒数。

    Returns:
        正文文本，截断到 MAX_OUTPUT。
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
    except Exception as e:  # noqa: BLE001
        return f"打不开这个网页：{type(e).__name__} {e}"

    text = _TAG.sub(" ", html)
    text = _HTML.sub(" ", text)
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_OUTPUT] or "（页面没有可读的正文）"


async def web_search(query: str, client, model: str, timeout: float = 35.0) -> str:
    """联网搜一下。

    走 OpenRouter 的 ``:online`` 变体——模型名加个后缀就行，不用额外的搜索 API key。
    比普通调用贵得多（一次约合七厘钱），所以只在这个工具里用。

    Args:
        query: 要搜什么。
        client: AsyncOpenAI 兼容客户端。
        model: 基础模型名，函数内部会加 ``:online``。
        timeout: 超时秒数。

    Returns:
        简要回答。
    """
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=f"{model}:online",
                messages=[
                    {
                        "role": "system",
                        "content": "用中文简要回答，只说查到的事实和数字，不要列链接。",
                    },
                    {"role": "user", "content": query},
                ],
                max_tokens=400,
            ),
            timeout=timeout,
        )
        return (resp.choices[0].message.content or "").strip() or "没搜到。"
    except Exception as e:  # noqa: BLE001
        return f"搜索失败：{type(e).__name__} {e}"
