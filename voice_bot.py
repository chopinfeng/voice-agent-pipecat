"""中文语音 agent：本地 FunASR 听写 + OpenRouter 大模型 + 本地 Piper 合成。

只有 LLM 走云端，识别和合成都在本机，因此 OpenRouter 的 key 是唯一需要的凭据。

端到端延迟（realtime_replay.py 实测 5 轮中位，真 VAD、挂着全套工具）：
    VAD 判定说完                          0.5s（从真的说完算起）
    听写  FunASR SenseVoiceSmall          0.73s
    轮次判定（与听写并行，已被吸收）           0.85s
    LLM   qwen3-max 首 token              2.82s
    回答开始合成                           4.43s
用户说完到听见声音 1.10 秒——那是填充语顶上的，真正的回答在 4.4 秒左右接上。

**测延迟前先确认机器是干净的。**残留的 voice_bot 进程会把这些数字放大几十倍
（实测负载 44 时听写量出 68.9 秒、顺序都乱了）。清理用 ``pkill -f voice_bot.py``，
别用按端口杀——没绑上端口的僵尸进程会漏网，而它们照样占着模型和内存。

后台还挂了一个 agent worker：问到项目里的代码时，语音这边派单出去、继续说话，
不阻塞对话；agent 在自己的循环里翻文件，查完了再念结果。设 AGENT=0 可以关掉。

运行：
    uv run --project pipecat python voice_bot.py -t webrtc
    PROJECT_PATH=/path/to/repo uv run --project pipecat python voice_bot.py -t webrtc

然后打开 http://localhost:7860/client/ 点 Connect 说话。
"""

import asyncio
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pipecat.adapters.schemas.direct_function import tool_options
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMMessagesAppendFrame,
    LLMRunFrame,
    LLMSetToolsFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.job_context import JobError, JobEvent
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

import memory
import progress_api
from addressee import build_gate
from agent_worker import build_agent_worker
from diarize import build_diarizer
from filler import DEFAULT_PHRASES, FillerSpeech, synthesize_clips
from filters import (
    CollapseUserTurns,
    EmptyTranscriptionFilter,
    RememberInterrupted,
    SkipFilter,
    StripEmotionMarks,
    TermCorrection,
)
from latency_observer import LatencyObserver

load_dotenv(Path(__file__).parent / ".env", override=True)

MODEL_DIR = Path(__file__).parent / "piper-voices"
# 对话模型只需要说得快、中文自然；需要翻代码或联网的活儿一律派给后台 agent，
# 那边用 AGENT_MODEL 单独配，追的是质量不是速度。
#
# **只用中国开源模型**：当前 key 上 Anthropic / Google / OpenAI 全返回 403
# 「违反 provider 服务条款」。
#
# **选型要两步走，只看延迟会选错。**先量带工具的首 token 和尾巴筛掉慢的，
# 再拿五个真实场景验工具协议对不对——快而不能用比慢更糟：
#
#   模型              带工具首token   工具协议
#   qwen3-next-80b       1.93s      **把工具调用当正文吐出来**，那串 JSON 会被念给用户
#   deepseek 官方直连     0.62s      协议对，但天气和记忆全派给 ask_project（选错工具）
#   qwen3-30b            3.37s      同样泄漏 JSON
#   qwen3-max            3.48s      五项全过  ← 唯一可用
#
# 光看延迟表会选 qwen3-next-80b，它快 1.5 秒但会让用户听到
# `"arguments": {"query": ...}`。功能检查那一步是这次选型的决定性环节。
#
# glm-4.6 和 ling-3.0-flash 更早出局：推理型，400 个 token 预算内**一个正文字都不吐**。
#
# **瓶颈是网关不是模型**：同样的活儿直连 DeepSeek 官方只要 0.62 秒，走 OpenRouter
# 要 2.8-3.6 秒。哪天找到「直连 + 工具选择正确」的组合，对话延迟能掉进一秒内。
LLM_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3-max")
TTS_VOICE = os.getenv("PIPER_VOICE", "zh_CN-huayan-medium")
# 合成引擎。Kokoro 试过了，**两头都不占**：首帧慢一个数量级（对比见下面构造处），
# 中文听感也没比 Piper 好——它的强项是英文，中文音色是附带的。所以留在 piper。
# 想再听一遍：TTS_ENGINE=kokoro KOKORO_VOICE=zf_xiaoxiao。
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")
# Kokoro 的中文音色：zf_ 开头是女声，zm_ 是男声。
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "zf_xiaoxiao")
# 轮次窗口跟听写是**并行**的，都从 VAD 判定说完开始计时，所以它只在超过听写耗时的
# 部分才真的加延迟。实测（realtime_replay.py，真 VAD，各 4 轮中位）：窗口 0.8 秒时
# 轮次判定 0.82 秒、0.4 秒时 0.71 秒、0.2 秒时 0.82 秒——最后那档精确等于听写耗时，
# 说明窗口已经完全被吸收，再降一点用没有。0.4 是拐点。
#
# 降它不会让句子被切碎：切段是 VAD_STOP_SECS 管的，这个窗口只管切好的段攒成几轮。
SPEECH_TIMEOUT = float(os.getenv("SPEECH_TIMEOUT", "0.4"))
# VAD 判定说话停止的静音时长。它决定听写切几段——静音一超过它，一句话就会被切开
# 分别转写，模型收到的是碎片。轮次窗口只管这些段攒成几轮，改它救不了碎片。
# vad_probe.py 实测（带随机留白的长句）：0.2 秒切 3.0 段、0.8 秒切 2.3 段、
# 1.2 秒切 2.0 段，代价是说完到听写收齐从 -0.1 秒涨到 0.9 秒。0.8 是折中点。
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))
# 设 STT=paraformer 换成带热词的 Paraformer：中文专有名词准不少（「填充语」不再被
# 听成「填充与」），代价是模型更大、单次慢 0.15 秒。默认仍是 SenseVoice——
# 全套场景是在它上面验过的。
STT_ENGINE = os.getenv("STT", "sensevoice")
# 设 FILLER_DELAY=0 关掉填充语。
FILLER_DELAY = float(os.getenv("FILLER_DELAY", "0.25"))
# 设 AGENT=0 关掉后台 agent worker。
AGENT_ENABLED = os.getenv("AGENT", "1") != "0"
AGENT_NAME = "project-agent"
# 两次进度播报之间至少隔这么久，避免碎碎念。
PROGRESS_MIN_GAP = float(os.getenv("PROGRESS_MIN_GAP", "12"))
# 联网查询的超时。搜索本身要花时间，给得比普通调用宽。
WEB_TIMEOUT = float(os.getenv("WEB_TIMEOUT", "35"))
# 设 DIRECT_DELIVER=1 切回「结果绕过模型直接念」的旧路径。见 _deliver 的说明——
# 留着是为了能在同一场手测里 A/B，不是为了长期两条路都养着。
DIRECT_DELIVER = os.getenv("DIRECT_DELIVER", "0") != "0"

SYSTEM_INSTRUCTION = (
    # 不写这段，用户闲聊时问起"你是本地还是云端的"，模型会张口就来说自己跑在云上、
    # 语音也是云端合成的——全说反了。
    "关于你自己：你的语音识别用本机的 FunASR，语音合成用本机的 Piper 中文模型，"
    "只有大模型这一部分走 OpenRouter 云端接口。用户问起你怎么跑的，照这个说。"
    "你是一个中文语音助手。你的回答会被朗读出来，所以不要使用表情符号、"
    "项目符号或任何无法朗读的格式，数字和单位都写成口语说法。"
    "回答控制在一到两句话以内，第一句尽量短，这样用户能更快听到声音。"
    # 早先这里写的是「你没有查实时信息的能力」，那是为了防它编造时间。结果一刀切
    # 太狠——问天气问股价全被拒答。改成给它一把工具，同时保留「不许编」这条。
    "遇到天气、股价、汇率、新闻、赛事比分这类需要联网才知道的事，调 search_web。"
    "工具查不到就如实说查不到，**任何情况下都不许自己编一个数字或事实出来**。"
    "你确实不知道现在几点——系统没给你时钟，被问到就直说。"
    "遇到关于本地项目的问题，调用 ask_project 工具——代码、文件、目录结构、依赖、测试，"
    "**需要动手算的题目也派给它**（比如「第几个斐波那契数是多少」这种大数计算）——"
    "它手里有 shell 能跑 python，你算不出来不等于办不到，不许直接说自己没这个能力。"
    "**以及这个项目里某个功能到底是怎么实现的（哪怕问的是你自己的某个能力，"
    "比如填充语、延迟优化、听写怎么做的），也要去看代码，不许凭常识编**。"
    # dev 是唯一会真的改文件的类型，且只有用户明确要求改代码时才用——听写一旦
    # 听岔，「查一下」和「改一下」是完全不同的两件事，不能靠猜。
    "用户明确要求**改代码、写代码、修复某个问题、加个功能、跑测试**这类要动手改的"
    "事，才把 kind 定为 dev；只是问问题、查资料、想了解某个功能怎么实现的，用"
    "codebase 或 general，**不要因为话题跟代码有关就顺手定成 dev**。"
    "后台有一个 agent 会去翻文件，它可能要跑几十秒。它返回结果后，用口语把要点讲给用户。"
    "任务跑的过程中用户可以照常聊别的，你正常回答就行。"
    "如果后台已经有任务在跑，用户又问起它的进展（比如「查得怎么样」「好了吗」「还要多久」），"
    "调 check_progress，绝对不要重新派一遍同样的任务。"
    "任务跑着的时候用户提出**另一个不同的**问题，用 ask_another 排队。"
    "用户说不查了、算了、停下，调 cancel_task；只想停其中一个就把那件事的关键词传给它。"
    # 多人在场。前缀是系统按声纹加的，模型看得见但绝不能念出来——这个项目里
    # 内部标注被当成内容复述已经发生过两次（合并提示、调度理由）。
    "\n每句话开头的 [主用户]、[说话人2] 这类前缀是系统按声纹加的标注，"
    "**它不是用户说的内容，任何情况下都不要念出来、不要提到它**。"
    "[主用户] 是这台设备的主人，你主要为他服务。"
    "现场可能有别人在说话，他们的话也会被听进来。判断一句话是不是在跟你说："
    "带称呼、提要求、问问题、接着你刚才的话往下说，就是在跟你说；"
    "两个人在聊他们自己的事、议论第三方、说的内容跟你完全无关，就不是。"
    "**确定不是在跟你说话时，只回复 [skip] 四个字符，不要有任何别的输出**——"
    "系统会让你保持安静。拿不准的时候正常回答，别滥用 [skip]。"
    # 长期记忆。写进存档而不是靠上下文记——上下文会被打断取消、被合并改写、
    # 被长度挤掉，长期事实放那儿等于没放。
    "\n用户交代**对以后也成立**的事时（住哪、怎么称呼他、喜欢什么样的回答），"
    "调 remember_this 存下来；他说别记了就调 forget_this。"
    "一次性的对话内容不要存。"
)

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


# 正在跑的后台任务，按 job_id 索引：用户可以在一个任务跑着的时候再派一个，
# 查进度和取消都要能分清是哪一个，所以不能只留一个槽位。
_active: dict[str, dict] = {}


def _searching_line(query: str) -> str:
    """联网前先说的那句话。

    带上查什么，用户才知道它有没有听对——「我查一下天津天气」比干巴巴一句「稍等」
    有用得多，听错了可以当场纠正，不用等十秒钟拿到一个答非所问的结果。
    """
    topic = query.strip()[:14]
    return f"我查一下{topic}啊。"


@tool_options(cancel_on_interruption=False, timeout_secs=40)
async def search_web(params: FunctionCallParams, query: str):
    """查需要联网才知道的实时信息：天气、股价、汇率、新闻、赛事比分等。

    Args:
        query (str): 要查什么，写成完整的一句话，比如「北京今天天气」。
    """
    logger.info(f"联网查询：{query}")
    job_id = f"search-{params.tool_call_id}"
    # 登记。用户眼里搜索和后台任务是一回事——都是「我让它去办、还没办完的事」。
    # 只登记 agent 任务的话，用户问「现在有什么在跑」会被答「没有」，而当时其实有个
    # 搜索正在飞（实测发生过）。
    progress_api.start(job_id, query, kind="search")
    _active[job_id] = {"question": query, "kind": "search"}
    # 先出声再去查。联网要五到十秒，这期间一声不吭的话用户等两秒就会再问一遍，
    # 而再问就是一次打断。
    await params.llm.queue_frame(TTSSpeakFrame(_searching_line(query)))

    online = f"{LLM_MODEL}:online"
    try:
        task = asyncio.create_task(_web_query(online, query))
        _active[job_id]["task"] = task
        answer = await asyncio.wait_for(asyncio.shield(task), timeout=WEB_TIMEOUT)
    except asyncio.CancelledError:
        progress_api.finish(job_id, state="cancelled")
        _active.pop(job_id, None)
        await params.result_callback("[这条查询已被用户取消，不用再提。]")
        return
    except (Exception, asyncio.TimeoutError) as e:  # noqa: BLE001
        logger.warning(f"联网查询失败：{type(e).__name__} {e}")
        progress_api.finish(job_id, state="error", answer=str(e))
        _active.pop(job_id, None)
        await _deliver(params, job_id, "网上没查到，可能是网络不通，你稍后再问问。")
        return
    progress_api.finish(job_id, state="done", answer=answer)
    _active.pop(job_id, None)
    await _deliver(params, job_id, answer)


async def _deliver(params: FunctionCallParams, job_id: str, answer: str) -> None:
    """把办完的结果交给模型，由它讲给用户。

    **这里之前的诊断是错的，值得记一笔。**原先写的是「用户在等待期间随口说一句就是
    一次打断，那一轮被取消，几十秒的工作成果就此消失」，据此绕过 ``result_callback``
    直接推 ``TTSSpeakFrame``。对着 pipecat 源码核过之后，实际行为是：

    * 结果**不会丢**。``_handle_function_call_finished`` 无条件把它作为一条
      developer 消息写进上下文（``async_tool_messages`` 那套异步工具协议），
      取消也走同一条路。
    * 丢的只是**主动出声的那一次推理**：
      ``if run_llm and not self._user_speaking`` —— 结果落地时用户恰好在说话，
      那一轮就不跑，结果躺在上下文里等下一次自然轮次。

    而 bot 在说话时上游是留了补跑标记的（``_push_context_on_bot_stopped_speaking``），
    用户在说话时没留。这个不对称由 ``filters.RememberInterrupted`` 补上了，所以现在
    走正路就行：结果交给模型，它用自己的口吻讲出来。

    这么改不只是少一处 hack。直接 TTS 那条路上，agent 的原文是**没经过对话模型润色**
    的，跟系统提示里「一到两句话、第一句尽量短」那些约束对不上；而且还要额外塞一条
    「已经念过了别复述」的假消息去堵模型的嘴，那条消息本身也在污染上下文。

    代价是多一轮 LLM（约 1.5 秒）才出声。所以旧路径留在 ``DIRECT_DELIVER=1`` 后面，
    同一场手测里可以来回切着听哪个好。
    """
    logger.info(f"投递结果（{job_id}）：{answer[:40]}")
    if DIRECT_DELIVER:
        await params.llm.queue_frame(TTSSpeakFrame(answer))
        await params.result_callback(
            f"[系统已经把这个结果念给用户了，不要复述：{answer[:120]}]"
        )
        return
    await params.result_callback(answer)


async def _web_query(model: str, query: str) -> str:
    """打一次联网查询，把结果收拾成适合朗读的样子。"""
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "用一两句中文口语回答，只说查到的事实和数字。"
                        "不要列参考链接、不要 markdown、不要脚注标记——"
                        "这些会被原样念出来。查不到就直说查不到。"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip()
    finally:
        await client.close()
    return clean_for_speech(text)


# 联网结果里的出处标注：markdown 链接、括号里的网址、以及**光秃秃的域名**。
# 最后一种最容易漏——实测念出来是「weather.com.cnnmc.cn」这种，因为模型把两个
# 来源直接连在了一起。
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BRACKET_URL = re.compile(r"[\[\(](?:https?://|www\.)[^\]\)]*[\]\)]")
_BARE_DOMAIN = re.compile(
    r"[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*"
    r"\.(?:com|cn|org|net|gov|edu|io|ai|co|info)(?:\.[a-z]{2})?",
)


def clean_for_speech(text: str) -> str:
    """把联网结果收拾成能念的样子。

    Args:
        text: 模型返回的原文。

    Returns:
        去掉出处标注、可以直接朗读的文本。
    """
    text = _MD_LINK.sub(r"\1", text)
    text = _BRACKET_URL.sub("", text)
    text = _BARE_DOMAIN.sub("", text)
    # 清完常留下孤零零的标点和多余空格。
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([，。、！？])", r"\1", text)
    text = re.sub(r"[，、]\s*。", "。", text)
    return text.strip(" ，、") or "没查到。"


@tool_options(cancel_on_interruption=False, timeout_secs=180)
async def ask_project(params: FunctionCallParams, question: str, kind: str = "general"):
    """交给后台 agent 去查或去算：本地项目的代码文件依赖测试，以及任何需要动手
    算一算、跑段代码才能得出的问题（大数计算、数列、统计、格式转换等）。

    agent 手里有 shell，可以跑 python。你自己心算不出来的题目派给它，不要硬答，
    也不要说自己没有这个能力。

    Args:
        question (str): 要查或要算的问题，写成完整的一句话。
        kind (str): 任务类型，决定后台用哪套提示和工具。可选：compute=要算的：大数、数列、统计、进制转换、日期推算；codebase=要查本地代码的：某个功能怎么实现的、文件在哪、依赖有哪些；research=要联网深入查的：需要看几个来源、比对之后才能回答；dev=要动手改的：写代码、改已有文件、跑测试，只有明确要求改代码才用这个；general=说不清属于哪类，或者要几种活一起干。
    """
    await _dispatch(params, question, kind)


@tool_options(cancel_on_interruption=False, timeout_secs=180)
async def ask_another(params: FunctionCallParams, question: str, kind: str = "general"):
    """后台已经有任务在跑时，再排一个**不同的**新问题，它会排在后面依次执行。

    只有当用户明确提出了一个新的、和正在查的那件事不同的问题时才用这个。
    用户只是催进度、问好了没、重复问同一件事，一律用 check_progress。

    Args:
        question (str): 新的、和在跑的那个不同的问题。
        kind (str): 任务类型，可选：compute=要算的：大数、数列、统计、进制转换、日期推算；codebase=要查本地代码的：某个功能怎么实现的、文件在哪、依赖有哪些；research=要联网深入查的：需要看几个来源、比对之后才能回答；dev=要动手改的：写代码、改已有文件、跑测试，只有明确要求改代码才用这个；general=说不清属于哪类，或者要几种活一起干。
    """
    await _dispatch(params, question, kind)


async def _dispatch(params: FunctionCallParams, question: str, kind: str = "general"):
    """把问题派给后台 agent，边等边报进度，结束后交结果。"""
    logger.info(f"派单给后台 agent（{kind}）：{question}")
    job_id = None
    try:
        # 派单是异步的：进入 with 块之后语音这边继续说话，agent 在另一个 worker 里跑，
        # 语音管线不会被几十秒的工具循环卡住。
        async with params.pipeline_worker.job(
            AGENT_NAME,
            name="ask",
            payload={"question": question, "kind": kind},
            timeout=180,
        ) as agent_job:
            job_id = agent_job.job_id
            _active[job_id] = {"job": agent_job, "question": question}
            # 工具调用要几十秒才返回，在那之前模型的上下文里看不到"有任务在跑"这件
            # 事——用户再问一句「查得怎么样」，它就会当成新请求再派一遍。光靠 prompt
            # 和上下文状态纠正不过来（试过），所以直接把 ask_project 从工具表里摘掉：
            # 任务跑着的时候它只剩查进度和取消两个选择。聚合器开了
            # add_tool_change_messages，会自动往上下文写一条工具变更说明。
            await params.llm.queue_frame(LLMSetToolsFrame(tools=BUSY_TOOLS))
            await params.llm.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {
                            "role": "developer",
                            "content": (
                                f"[状态] 后台任务已启动，正在查：{question}。"
                                "任务还没结束。用户再问它的进展时调 check_progress，"
                                "不要重复调 ask_project。"
                            ),
                        }
                    ],
                    run_llm=False,
                )
            )
            await params.llm.queue_frame(
                LLMMessagesAppendFrame(
                    messages=[
                        {"role": "developer", "content": "告诉用户你正在查，让他稍等。"}
                    ],
                    run_llm=True,
                )
            )
            # 边等边收进度。agent 一步里可能并发调好几个工具，按步数节流会一次念出
            # 七八条，所以按时间间隔来：隔一段时间报一次当前在做什么，其余只记日志。
            last_spoken, last_text = 0.0, ""
            async for event in agent_job:
                if event.type != JobEvent.UPDATE:
                    continue
                data = event.data or {}
                spoken = data.get("spoken") or data.get("note", "")
                logger.info(f"后台进度 第{data.get('step', 0)}步：{data.get('note', '')}")
                now = time.monotonic()
                # urgent 是用户主动问出来的，要立刻播；自动进度按间隔节流。两者会撞在
                # 一起，所以内容一样就不重复念——否则同一句「跑了 88 秒」会连播两遍。
                fresh = spoken != last_text
                due = now - last_spoken >= PROGRESS_MIN_GAP
                if fresh and (data.get("urgent") or due):
                    last_spoken, last_text = now, spoken
                    await params.llm.queue_frame(TTSSpeakFrame(spoken))
        response = agent_job.response or {}
        answer = response.get("answer") or response.get("error", "没查到。")
    except JobError as e:
        # 用户中途喊停会走到这里。这是正常结局，不该往管线里推错误帧——
        # cancel_task 已经跟用户说过了，这里只需要安静收场。
        logger.info(f"后台任务提前结束：{e}")
        answer = ""
    finally:
        _active.pop(job_id, None)
        # 还有别的任务在跑就继续压着 ask_project，全部结束了才放开。
        await params.llm.queue_frame(
            LLMSetToolsFrame(tools=BUSY_TOOLS if _active else ALL_TOOLS)
        )
    if answer:
        # 跟联网查询同样的道理，而且这边丢掉的是几十秒甚至几分钟的工作。见 _deliver。
        await _deliver(params, job_id, answer)


async def _sync_memory(params: FunctionCallParams) -> None:
    """把存档里的记忆重新灌进上下文。

    系统提示是**建 LLM service 时拼一次**的（``run_bot`` 里的 ``memory.as_prompt()``），
    所以会话中途新记的事根本不在系统提示里——它只活在这一轮工具调用留下的历史里。

    而 ``memory.py`` 开头那段自己的论证正是「历史会被打断取消、被合并改写、被上下文
    长度挤掉，把长期事实放在那儿等于没放」。这个项目里 ``CollapseUserTurns`` 和打断
    收尾都会动那段历史。靠历史带记忆跟那段论证是自相矛盾的，重启后能记得、这一场
    反而可能记不住。

    所以每次增删之后补一条 developer 消息。``run_llm=False``：这只是把事实摆进去，
    该跟用户说什么由 ``result_callback`` 那一轮负责。

    同一场会话里多次增删会留下多条，最新的在最后。记忆增删是低频动作，先不做去重。
    """
    block = memory.as_prompt() or "（用户目前没有让你长期记住的事。）"
    await params.llm.queue_frame(
        LLMMessagesAppendFrame(
            messages=[{"role": "developer", "content": f"[长期记忆·最新] {block}"}],
            run_llm=False,
        )
    )


@tool_options(cancel_on_interruption=False)
async def remember_this(params: FunctionCallParams, topic: str, content: str):
    """把用户交代的、以后还该记得的事存下来，跨会话有效。

    用户说「我住在北京」「以后回答简短点」「叫我老张」这类**对以后也成立**的事时调它。
    不要拿它记一次性的对话内容——那是日志的事。

    Args:
        topic (str): 归类用的短词，比如「住址」「称呼」「回答风格」。同一个词再记会
            覆盖旧的，用户改主意时不会留下两条打架的记录。
        content (str): 要记的事，一句话。
    """
    said = memory.remember(topic, content)
    await _sync_memory(params)
    await params.result_callback(said)


@tool_options(cancel_on_interruption=False)
async def forget_this(params: FunctionCallParams, topic: str):
    """忘掉之前记住的某件事。用户说「别记着我住哪了」时调。

    Args:
        topic (str): 要忘掉的那类事，比如「住址」。
    """
    said = memory.forget(topic)
    await _sync_memory(params)
    await params.result_callback(said)


@tool_options(cancel_on_interruption=False)
async def check_progress(params: FunctionCallParams):
    """查询已经在跑的那个后台任务进行到哪一步了。

    用户问「查得怎么样了」「好了吗」「还要多久」「有进展吗」时用这个。它只读进度，
    不会重新派任务——同一个问题不要再调 ask_project。
    """
    tasks = progress_api.snapshot()
    if not tasks:
        await params.result_callback("现在没有在跑的后台任务。")
        return
    # 顺带向 worker 要一次刷新，它会以 urgent 优先级立刻回过来。只有 agent 任务能问
    # ——搜索不是 worker 上的 job，问它会打空。
    for job_id, item in list(_active.items()):
        if item.get("kind") != "search":
            await params.pipeline_worker.request_job_update(job_id, AGENT_NAME)
    lines = [
        f"{t['question'][:20]}：{progress_api.REGISTRY[t['job_id']].spoken()}"
        for t in tasks[:3]
        if t["job_id"] in progress_api.REGISTRY
    ]
    await params.result_callback("；".join(lines) or "刚开始查。")


@tool_options(cancel_on_interruption=False)
async def cancel_task(params: FunctionCallParams, which: str = ""):
    """取消后台任务。

    用户说「算了」「不用查了」「停下」时取消全部。用户只想停其中一个（「把查依赖
    那个停掉，另一个继续」），把那件事的关键词传进 ``which``。

    Args:
        which (str): 要停哪个任务的关键词，比如「依赖」「架构」。留空表示全停。
    """
    if not _active:
        await params.result_callback("现在没有在跑的后台任务。")
        return

    targets = list(_active)
    if which:
        # 按关键词挑：用户是按内容指认任务的，不会报 job_id。
        key = which.strip()
        matched = [
            job_id for job_id, item in _active.items() if key and key in item["question"]
        ]
        if not matched:
            running = "、".join(item["question"][:16] for item in _active.values())
            await params.result_callback(
                f"没找到跟「{which}」对得上的任务。在跑的是：{running}。"
            )
            return
        targets = matched

    stopped = [_active[job_id]["question"][:16] for job_id in targets]
    for job_id in targets:
        item = _active.get(job_id, {})
        # 两种活儿停法不同：agent 任务是总线上的 job，搜索只是本地一个协程。
        # 之前只会停前者，用户说「不看演唱会门票了」时那个搜索照样在跑。
        if item.get("kind") == "search":
            task = item.get("task")
            if task and not task.done():
                task.cancel()
            progress_api.finish(job_id, state="cancelled")
        else:
            await params.pipeline_worker.cancel_job_group(job_id, reason="用户取消")
        _active.pop(job_id, None)
    left = len(_active)
    tail = f"，还有{left}个在跑" if left else "，没有别的在跑了"
    await params.result_callback(f"好的，停了：{'、'.join(stopped)}{tail}。")


# 平时全都给；后台任务跑着的时候摘掉 ask_project，逼模型走查进度或取消这两条路。
# 平时给 ask_project。有任务在跑时换成 BUSY_TOOLS：ask_project 被摘掉（防止用户
# 催进度时模型重新派一遍同样的任务），但留一个 ask_another——用户提出的确实是**新**
# 问题时还有路可走，不至于被一句"现在没法查"堵死。
ALL_TOOLS = [ask_project, search_web, check_progress, cancel_task,
             remember_this, forget_this]
BUSY_TOOLS = [ask_another, search_web, check_progress, cancel_task,
              remember_this, forget_this]


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    # 本地：SenseVoiceSmall，非自回归推理，中文识别快且准。
    # 首次运行会下载模型到 ~/.cache/modelscope。
    if STT_ENGINE == "streaming":
        from streaming_stt import FunASRStreamingSTTService

        stt = FunASRStreamingSTTService()
    elif STT_ENGINE == "paraformer":
        from paraformer_stt import ParaformerSTTService, project_hotwords

        stt = ParaformerSTTService(
            hotwords=project_hotwords(Path(os.getenv("PROJECT_PATH", ".")))
        )
    else:
        stt = FunASRSTTService(
            settings=FunASRSTTService.Settings(language=Language.ZH)
        )

    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model=LLM_MODEL,
            # 每次连接时把存档里的记忆拼进提示。放系统提示而不是靠对话历史带——
            # 历史会被打断取消、被合并改写、被长度挤掉，长期事实放那儿等于没放。
            system_instruction=SYSTEM_INSTRUCTION + memory.as_prompt(),
            # 让 OpenRouter 优先选延迟最低的 provider。
            extra={"extra_body": {"provider": {"sort": "latency"}}},
        ),
    )

    # 两个都在本地跑，差别是音质换延迟，实测首帧（预热后）：
    #                        「好的。」   「好的，我帮你看一下。」  整句
    #   Piper huayan-medium     44ms          125ms            221ms
    #   Kokoro zf_xiaoxiao     460ms         1083ms           2597ms
    # 真实链路是按句合成的，所以短句那两栏才是实际感受到的代价。
    if TTS_ENGINE == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        tts = KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=KOKORO_VOICE, language=Language.ZH
            )
        )
    else:
        tts = PiperTTSService(
            download_dir=MODEL_DIR,
            settings=PiperTTSService.Settings(voice=TTS_VOICE),
        )

    context = LLMContext(tools=ALL_TOOLS if AGENT_ENABLED else None)
    user_params = LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS))
    )
    if SPEECH_TIMEOUT > 0:
        # 默认的 smart-turn 轮次判定在离线回放里判不出用户说完，一路等到
        # user_turn_stop_timeout（5 秒）才放行，端到端要 8 秒。换成固定停顿窗口后
        # 降到 3.3 秒。窗口就是用户停顿多久算说完，调小更快但更容易在句中被抢话。
        # 设 SPEECH_TIMEOUT=0 可以切回 smart-turn 自己对比。
        user_params.user_turn_strategies = UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=SPEECH_TIMEOUT)]
        )
    pair = LLMContextAggregatorPair(
        context,
        user_params=user_params,
        # 工具表中途变化时，往上下文写一条说明，模型才不会去调已经被摘掉的工具。
        add_tool_change_messages=True,
    )
    user_aggregator = pair.user()
    # 助手那半自己建，不用 pair 给的那个：默认实现被打断时会把说了一半的回复整个
    # 丢掉，上下文里于是连着好几条用户发言都没有助手消息，模型不记得自己说过什么
    # 就一直重复同一个答案——实测一段对话末尾连着四轮都是重复。
    assistant_aggregator = RememberInterrupted(
        context,
        params=LLMAssistantAggregatorParams(add_tool_change_messages=True),
        _paired_user_aggregator=user_aggregator,
    )

    stages = [
        transport.input(),
        stt,
        # SenseVoice 会往转写里塞情感表情，说完那段静音又会被听成一个孤零零的
        # 句号——前者会被念出来，后者会白打一次 LLM。
        StripEmotionMarks(),
        TermCorrection(),
        EmptyTranscriptionFilter(),
        # 判断每句是不是冲着助手说的。默认只打分记日志，不拦——真实麦克风下的
        # RMS 分布跟合成音频不一样，阈值得先看 logs/addressee.jsonl 再定。
        # 默认只观测不拦。音量阈值是绝对值，真实麦克风比合成音频低一个数量级，
        # 没在本机标过就开会把用户自己的话当背景丢掉（实测 11 句拦 8 句）。
        build_gate(),
        # 声纹分说话人，给每句话加 [主用户] / [说话人2] 前缀。只标注不拦截——
        # 判错的代价是一个标签，而不是吞掉用户的话。回不回由模型自己定。
        build_diarizer(),
        user_aggregator,
        # 连着被打断几次后，上下文尾部会攒下几条没人回答的用户发言，模型会当成
        # 并列请求一起消化。并成一条，只让最后一句作为要回应的请求。
        CollapseUserTurns(),
        llm,
        # 模型判定「这句不是对我说的」时输出 [skip]，在这里拦掉不进 TTS。
        SkipFilter(),
        tts,
    ]
    if FILLER_DELAY > 0:
        # 等待期间先应一声，把网络和模型那一段藏起来。必须放在 TTS 之后，否则
        # 填充语会被助手聚合器当成模型的回答写进上下文。
        # 引擎跟正文保持一致，否则一轮里会出现两个嗓子。填充语是启动时合成一次
        # 缓存起来的，所以用慢引擎也不影响响应速度。
        clips, rate = synthesize_clips(
            download_dir=MODEL_DIR,
            voice=KOKORO_VOICE if TTS_ENGINE == "kokoro" else TTS_VOICE,
            phrases=DEFAULT_PHRASES,
            engine=TTS_ENGINE,
        )
        stages.append(FillerSpeech(clips=clips, sample_rate=rate, delay=FILLER_DELAY))
    stages += [transport.output(), assistant_aggregator]
    pipeline = Pipeline(stages)

    worker = PipelineWorker(
        pipeline,
        name="voice",
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        # 每轮对话结束时把逐环节耗时打进日志，并追加到 logs/latency.jsonl。
        observers=[LatencyObserver()],
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"客户端已连接，使用 {LLM_MODEL} 开场")
        context.add_message({"role": "user", "content": "请用一句话跟我打个招呼。"})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("客户端已断开")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    workers = [worker]
    if AGENT_ENABLED:
        # 后台 agent 挂在同一条总线上，和语音管线并排跑。
        workers.append(build_agent_worker(AGENT_NAME))
    await runner.add_workers(*workers)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import app, main

    # 后台任务的进度除了播给用户听，也开一个 HTTP 口子给外部程序读：
    #   GET /api/progress            全部任务
    #   GET /api/progress/{job_id}   单个任务
    progress_api.install_routes(app)

    main()
