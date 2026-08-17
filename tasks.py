"""任务类型：派单时就定死干哪种活，别让 agent 自己猜。

实测过一次很说明问题的失败：让 agent 算「第 7856 个斐波那契数」，它 1 轮、0 次工具
调用，纯靠推理答了个错的。沙箱是放行 ``python3`` 的，工具也在手边——**卡点在系统提示
第一句写的是「你在回答关于一个代码项目的问题」**，算数题不在这个框里，模型就凭脑子
答了。把那句话换成「需要精确计算就写 python 跑出来」之后，它立刻 Bash 跑脚本，答案正确。

结论是任务类型必须**在派单时定下来**，不能指望一个通用提示覆盖所有活。语音侧的模型
本来就在选工具，顺手标一下类型即可，比在 agent 内部再做一次委派判断可靠——那一跳会
犯的错跟上面完全一样，只是挪上了一层。

每种类型三样东西不同：**提示的第一句**（决定它往哪个方向使劲）、**工具集**（给不该用的
工具只会诱导它走弯路）、**轮数预算**（算一道题两三轮就够，翻代码要十几轮）。

``SUBAGENTS`` 是留给「一个任务里要干几种活」的：那种情况才值得让主 agent 委派，
比如「查一下这个库的最新用法，然后对照我们的代码看要改哪里」。默认路径不走它。
"""

from dataclasses import dataclass, field

# 朗读约束每种任务都要，单独抽出来免得各写一遍写漏。
SPEAK = (
    "回答会被朗读出来，用口语化的中文，不要 markdown、不要代码块、不要列表符号，"
    "文件名和路径念出来即可。回答控制在三句话以内，抓住重点。"
    "数字太长就说位数和开头结尾，别把几百位数字全念出来。"
)


@dataclass
class Kind:
    """一类任务的提示、工具和预算。"""

    name: str
    desc: str
    prompt: str
    tools: list[str]
    max_turns: int
    # 自带工具循环那条路用的工具名（tools.py 里的），跟 Claude 侧的名字不一样。
    builtin_tools: list[str] = field(default_factory=list)


COMPUTE = Kind(
    name="compute",
    desc="要算的：大数、数列、统计、进制转换、日期推算",
    prompt=(
        "你要算一道题。**一律写 python 用 Bash 跑出来拿结果，不许凭推理或心算给答案**"
        "——你的推理在大数上不可靠，而跑一次脚本是确定的。\n"
        "一次写完整的脚本，别分几步试探。算完直接报结果。\n" + SPEAK
    ),
    tools=["Bash"],
    max_turns=6,
    builtin_tools=["bash"],
)

CODEBASE = Kind(
    name="codebase",
    desc="要查本地代码的：某个功能怎么实现的、文件在哪、依赖有哪些",
    prompt=(
        "你在回答关于一个代码项目的问题，项目在 {root}，所有文件都在这个目录下。\n"
        "用 Read 读文件、Glob 找文件、Grep 搜内容、Bash 跑只读命令。"
        "先把相关文件真正读一遍再回答，不要只看文件名猜。\n"
        # Glob 递归这件事必须讲明白，否则会把 vendored 的第三方库整棵树捞上来，
        # 实测据此答出过「根目录下没有 python 文件」这种自相矛盾的话。
        "注意 Glob 是**递归**的；只想看某一层要用 Bash 跑 find 加 -maxdepth 1。"
        "子目录里可能有第三方库的完整源码，动辄上万个文件，别把它们算进来。\n" + SPEAK
    ),
    tools=["Read", "Glob", "Grep", "Bash"],
    max_turns=20,
    builtin_tools=["bash"],
)

RESEARCH = Kind(
    name="research",
    desc="要联网深入查的：需要看几个来源、比对之后才能回答",
    prompt=(
        "你要联网查清楚一件事。\n"
        # 后端之间联网能力差得远：Claude 侧有原生 WebSearch，而 DeepSeek Harness
        # 的 headless profile 一个联网工具都没有（只有 bash/fs/fs-search）。所以两条
        # 路都给出来，让它按手里有什么选——不给第二条的话，没有 WebSearch 的后端
        # 只会凭记忆答，还会编一个「搜索没配置」的解释。
        "**有 WebSearch 工具就用它**，拿到具体网址再用 WebFetch 看详情；"
        "**没有的话，用 Bash 跑 `python3 websearch.py \"要查什么\"`**，"
        "那个脚本就在工作目录里，会返回带来源的结果。\n"
        "**查到什么说什么，查不到就说查不到，绝不许凭记忆编数字或事实。**\n"
        "同一个意思不要反复换关键词搜，找到可信来源就停。\n" + SPEAK
    ),
    tools=["WebSearch", "WebFetch"],
    max_turns=10,
    builtin_tools=["search", "browse", "bash"],
)

GENERAL = Kind(
    name="general",
    desc="说不清属于哪类，或者要几种活一起干",
    prompt=(
        "你是一个助手，手里有 shell、文件工具和联网搜索，项目在 {root}。\n"
        "**需要精确计算的一律写 python 用 Bash 跑，不要凭推理给答案。**"
        "问代码就读文件，问外面的事就用 WebSearch。\n" + SPEAK
    ),
    tools=["Read", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"],
    max_turns=20,
    builtin_tools=["bash", "search", "browse"],
)

KINDS = {k.name: k for k in (COMPUTE, CODEBASE, RESEARCH, GENERAL)}


def pick(name: str | None) -> Kind:
    """按名字取任务类型，认不出来就用通用的。"""
    return KINDS.get((name or "").strip().lower(), GENERAL)


def choices() -> str:
    """给语音侧模型看的类型说明，直接进工具参数描述。"""
    return "；".join(f"{k.name}={k.desc}" for k in KINDS.values())


# 留给「一个任务里要干几种活」的委派路径。主 agent 用 Task 工具派给它们，
# 每个有自己的提示和工具集，上下文也互相隔离——研究那一堆网页不会挤占算题的上下文。
# 默认不启用：多一跳委派就多一次判断失误的机会，而按类型直接路由没有这个风险。
def subagents() -> dict:
    """构造 SDK 的 subagent 定义表。"""
    from claude_agent_sdk import AgentDefinition

    return {
        k.name: AgentDefinition(
            description=k.desc,
            prompt=k.prompt,
            tools=k.tools,
            maxTurns=k.max_turns,
        )
        for k in (COMPUTE, CODEBASE, RESEARCH)
    }
