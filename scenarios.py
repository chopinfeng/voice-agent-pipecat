"""场景测试：后台 agent 长时间干活，同时用户还在持续说话。

这类场景光看"派单能不能返回结果"是测不出问题的——真正会出错的地方是**并发**：
任务跑着的时候用户又说了话，进度播报和正常对话抢同一条 TTS，用户中途改主意，
或者一口气派两个任务。每个场景都在完整的 `WorkerRunner`（语音管线 worker +
agent worker）上跑，用注入用户轮次的方式代替麦克风。

场景写成脚本：一串 `(等到第几秒, 用户说的话)`，加新场景就是加一条数据。

台词默认**用 Piper 合成成语音**再按 20 毫秒一帧喂进管线，走完整的
听写 → 上下文聚合 → 大模型 → 合成 链路。这样连 STT 会不会把「算了不用查了」听岔
都能一起测到——纯注入文本是测不出这一层的。用 `--text` 可以切回注入文本的快模式
（省掉合成和听写的时间，适合只验编排）。

运行：
    uv run --project pipecat python scenarios.py          # 全部，语音输入
    uv run --project pipecat python scenarios.py 闲聊      # 只跑名字含「闲聊」的
    uv run --project pipecat python scenarios.py --text    # 注入文本的快模式
    uv run --project pipecat python scenarios.py --list    # 只看有哪些场景
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.funasr.stt import FunASRSTTService
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)
os.environ.setdefault("PROJECT_PATH", str(HERE))

import voice_bot as V  # noqa: E402  — 复用 bot 里的工具定义和 system prompt
from agent_worker import build_agent_worker  # noqa: E402
from filler import synthesize_clips  # noqa: E402
from filters import (  # noqa: E402
    CollapseUserTurns,
    EmptyTranscriptionFilter,
    StripEmotionMarks,
    TermCorrection,
)

# 几个足够重的问题，用来制造几十秒量级的后台任务。
ARCH = "把这个项目里所有 python 文件都看一遍，说说整体架构是怎么组织的"
LATENCY = "翻一下延迟测试相关的代码，说说这个项目是怎么量延迟的"
DEPS = "看看这个项目依赖了哪些第三方库，挑几个重要的说说"
# 这个足够重，实测要跑一分钟以上：延迟相关的文件有七八个，agent 得逐个读。
HEAVY = (
    "把延迟测试相关的代码全部读一遍，包括观察者、汇总报告、消融实验、实时回放这几个，"
    "详细说说每一个是干什么的、它们之间怎么配合"
)


class StubTTS(TTSService):
    """只吐静音的假 TTS。

    场景验的是编排不是音质，但 TTS 不能省：正常回答走 ``LLMTextFrame``、进度播报走
    ``TTSSpeakFrame``，只有经过 TTS 两者才会汇成同一股 ``TTSTextFrame``——那正是
    用户实际听到的顺序。
    """

    def __init__(self, **kwargs):
        super().__init__(
            settings=TTSSettings(model=None, voice=None, language=None), **kwargs
        )

    async def run_tts(self, text: str, context_id: str):
        yield TTSStartedFrame()
        yield TTSAudioRawFrame(
            audio=b"\x00" * 1600, sample_rate=self.sample_rate, num_channels=1
        )
        yield TTSStoppedFrame()


class Transcript(FrameProcessor):
    """把用户说的和助手说的按时间顺序记在一起。

    助手侧只看 ``TTSTextFrame``——那是真正会被念出来的内容，填充语和进度播报也在
    其中，正好用来检查两者有没有互相踩。
    """

    def __init__(self, share: "Transcript | None" = None):
        """Args:
        share: 共享另一份记录。听写帧到不了管线末尾（聚合器会吃掉），所以要在
            STT 后面再放一个探针，两处写进同一个列表才能排出完整时间线。
        """
        super().__init__()
        self.lines = share.lines if share else []
        self.t0 = share.t0 if share else time.perf_counter()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        now = time.perf_counter() - self.t0
        if isinstance(frame, TTSTextFrame) and frame.text.strip():
            self.lines.append((now, "助手", frame.text.strip()))
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            # 听写结果单独记一行：语音模式下这里能看出 STT 有没有把台词听岔。
            self.lines.append((now, "听写", frame.text.strip()))
        await self.push_frame(frame, direction)

    def note_user(self, text: str):
        self.lines.append((time.perf_counter() - self.t0, "用户", text))

    def dump(self, indent: str = "  "):
        marks = {"用户": ">>>", "听写": " ~ ", "助手": "   "}
        for at, who, text in sorted(self.lines, key=lambda r: r[0]):
            print(f"{indent}[{at:5.1f}s] {marks[who]} {text}")


@dataclass
class Scenario:
    """一条脚本化的对话。

    Parameters:
        name: 场景名，也是命令行筛选用的关键词。
        why: 这个场景想暴露什么问题。
        script: ``(等到第几秒, 用户说的话)``，秒数相对场景开始。
        total: 场景总时长，要留够让最后一个后台任务跑完。
    """

    name: str
    why: str
    script: list[tuple[float, str]]
    total: float


# 台词合成出来的音频缓存，同一句在多个场景里复用，省掉重复合成。
_VOICE_CACHE: dict[str, bytes] = {}
_VOICE_RATE = 0
CHUNK_SECS = 0.02


def say_as_audio(text: str) -> tuple[bytes, int]:
    """把一句台词合成成 16 位 PCM。

    用的就是 bot 自己那把 Piper 中文嗓子——测试里"用户"的声音和助手同源，音色
    单一，但对 STT 来说是干净可复现的输入，正好用来验听写这一环。
    """
    global _VOICE_RATE
    if text not in _VOICE_CACHE:
        clips, rate = synthesize_clips(
            download_dir=V.MODEL_DIR, voice=V.TTS_VOICE, phrases=[text]
        )
        _VOICE_CACHE[text] = clips[0]
        _VOICE_RATE = rate
    return _VOICE_CACHE[text], _VOICE_RATE


class Harness:
    """跑一条完整的双 worker 管线，可以随时让"用户"说话。

    Args:
        voice: True 走真语音（合成台词 → 听写 → 上下文），False 直接注入文本。
    """

    def __init__(self, voice: bool = True):
        self.voice = voice
        self.transcript = Transcript()
        self.context = LLMContext(tools=V.ALL_TOOLS)
        llm = OpenRouterLLMService(
            api_key=os.environ["OPENROUTER_API_KEY"],
            settings=OpenRouterLLMService.Settings(
                model=V.LLM_MODEL, system_instruction=V.SYSTEM_INSTRUCTION
            ),
        )
        # 助手聚合器不能省：工具里用 LLMSetToolsFrame 改工具表，那条帧要由聚合器
        # 落地并往上下文写变更说明，否则模型会去调已经被摘掉的工具。
        user_params = LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=V.VAD_STOP_SECS)),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=V.SPEECH_TIMEOUT
                    )
                ]
            ),
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            self.context, user_params=user_params, add_tool_change_messages=True
        )
        stages = []
        self.heard = Transcript(share=self.transcript)
        if voice:
            # 语音模式下把 bot 里听写那一段原样搬过来，包括那个空转写过滤器——
            # 台词末尾的静音同样会被听成一个孤零零的句号。
            stages += [
                FunASRSTTService(
                    settings=FunASRSTTService.Settings(language=Language.ZH)
                ),
                StripEmotionMarks(),
                TermCorrection(),
                EmptyTranscriptionFilter(),
                self.heard,
                user_aggregator,
            ]
        stages += [
            CollapseUserTurns(),
            llm,
            StubTTS(),
            self.transcript,
            assistant_aggregator,
        ]
        self.worker = PipelineWorker(
            Pipeline(stages),
            name="voice",
            params=PipelineParams(enable_metrics=True),
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self._task = None

    async def start(self):
        await self.runner.add_workers(self.worker, build_agent_worker(V.AGENT_NAME))
        self._task = asyncio.create_task(self.runner.run())
        await asyncio.sleep(1.5)
        now = time.perf_counter()
        self.transcript.t0 = self.heard.t0 = now

    async def say(self, text: str):
        """模拟用户说了一句话。"""
        self.transcript.note_user(text)
        if not self.voice:
            self.context.add_message({"role": "user", "content": text})
            # 文本模式下管线里没有用户聚合器，LLMRunFrame 没人翻译，直接注入上下文帧。
            await self.worker.queue_frames([LLMContextFrame(self.context)])
            return

        pcm, rate = say_as_audio(text)
        pcm += b"\x00" * (int(rate * 0.8) * 2)  # 尾部静音，让轮次判定收得住
        chunk = int(rate * CHUNK_SECS) * 2
        await self.worker.queue_frames([VADUserStartedSpeakingFrame()])
        for i in range(0, len(pcm), chunk):
            await self.worker.queue_frames(
                [
                    InputAudioRawFrame(
                        audio=pcm[i : i + chunk], sample_rate=rate, num_channels=1
                    )
                ]
            )
            # 按真实节奏喂，VAD 和轮次判定才拿得到跟麦克风一样的时序。
            await asyncio.sleep(CHUNK_SECS)
        await self.worker.queue_frames([VADUserStoppedSpeakingFrame()])

    async def run(self, scenario: Scenario):
        for at, text in scenario.script:
            now = time.perf_counter() - self.transcript.t0
            if at > now:
                await asyncio.sleep(at - now)
            await self.say(text)
        remaining = scenario.total - (time.perf_counter() - self.transcript.t0)
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def stop(self):
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)


SCENARIOS = [
    Scenario(
        name="进度自动播报",
        why="任务跑几十秒，助手不能全程沉默",
        script=[(0, ARCH)],
        total=80,
    ),
    Scenario(
        name="中途问进度",
        why="问进展要走 check_progress，不能重复派一遍任务",
        script=[(0, ARCH), (12, "查得怎么样了？")],
        total=80,
    ),
    Scenario(
        name="任务期间闲聊",
        why="闲聊必须立刻答上，不能等 agent 干完",
        script=[(0, ARCH), (10, "先别管那个，杭州今天适合出门吗？")],
        total=80,
    ),
    Scenario(
        name="中途取消",
        why="取消要真停住 agent，且不往管线推错误帧",
        script=[(0, ARCH), (15, "算了，不用查了。"), (25, "那你还在忙吗？")],
        total=50,
    ),
    Scenario(
        name="连派两个任务",
        why=(
            "agent 是 sequential 的，第二个会排队。语音侧只记了一个在跑的 job，"
            "两个任务同时在飞时查进度和取消会不会打到错的那个"
        ),
        script=[
            (0, ARCH),
            (20, "另外再帮我看看这个项目依赖了哪些第三方库。"),
            (45, "刚才那两件事都查得怎么样了？"),
            (75, "先把架构那个结果告诉我。"),
        ],
        total=150,
    ),
    Scenario(
        name="长对话穿插",
        why=(
            "任务跑着的时候连着聊四五轮不相干的，最后回来要结果——"
            "看上下文会不会被进度播报和闲聊搅乱，工具表有没有正确恢复"
        ),
        script=[
            (0, ARCH),
            (12, "对了，你是本地跑的还是云端的？"),
            (25, "那你说话的声音是怎么合成的？"),
            (40, "你觉得语音助手最难做的是哪一部分？"),
            (55, "说回来，刚才那个架构查完了吗？"),
            (85, LATENCY),
        ],
        total=170,
    ),
    Scenario(
        name="五分钟长跑",
        why=(
            "五分钟连续对话，中间挂一个跑一分钟以上的重任务：看长任务期间进度播报是否"
            "有实质内容（跑了多久、读了几个文件），穿插的闲聊和进度查询会不会互相干扰，"
            "任务结束后能不能自然接回来"
        ),
        script=[
            (0, HEAVY),
            (20, "这个要跑挺久吧，大概要多长时间？"),
            (45, "现在到哪一步了？"),
            (70, "你先说说，你觉得读代码这件事对语音助手来说难在哪？"),
            (100, "查得怎么样了？"),
            (130, "顺便问一下，你刚才说的填充语具体是怎么工作的？"),
            (165, "那个任务好了没？"),
            (200, "好的，把结果详细讲给我听。"),
            (250, "最后再帮我看看这个项目依赖了哪些第三方库。"),
        ],
        total=300,
    ),
    Scenario(
        name="打断助手",
        why=(
            "助手正播着长回答时用户插话。VAD 应该触发打断，助手立刻闭嘴转去应新问题，"
            "而不是把上一段念完——这是语音交互最基本的一条，之前一次都没测过"
        ),
        script=[
            (0, "简单说说这个项目是干什么的"),
            # 助手正在念开场回答时插进去。
            (8, "停一下，我改主意了，先告诉我现在几点"),
            (20, "行，那你继续说项目的事吧"),
        ],
        total=60,
    ),
    Scenario(
        name="连珠炮追问",
        why=(
            "用户不等回答就连说三句。输入会在管线里积压，看助手是把三句混成一团、"
            "还是逐条处理；也看后台任务会不会被重复派"
        ),
        script=[
            (0, "帮我看看这个项目有哪些文件"),
            (3, "另外架构是怎么样的"),
            (6, "还有依赖了什么库"),
            (10, "都查好了告诉我"),
        ],
        total=120,
    ),
    Scenario(
        name="越界访问",
        why=(
            "用语音让 agent 去读项目外的文件。工具的路径校验必须挡住，"
            "而且要挡得体面——助手应该说清楚读不了，不能把异常念给用户听"
        ),
        script=[
            (0, "帮我读一下上一级目录里的文件"),
            (25, "那读一下系统的 etc 目录下的 passwd 文件呢"),
            (55, "好吧，那还是看看项目自己的文件吧"),
        ],
        total=110,
    ),
    Scenario(
        name="十分钟马拉松",
        why=(
            "十分钟连续对话，把各种特殊说话方式一次压进来：问耗时预估、指代前文、"
            "说到一半改口、中英夹杂、数字单位、元请求（说慢点）、只回一个「嗯」、"
            "情绪化催促、并发任务里只取消其中一个、要求把结果压成三句。"
            "看的是长程一致性——上下文会不会串、工具选择会不会随轮次退化、"
            "任务状态会不会对不上"
        ),
        script=[
            (0, HEAVY),
            (18, "这个大概要跑多久啊"),
            (40, "你先说说，做语音助手最花时间的是哪一块"),
            (62, "刚才那个查得怎么样了"),
            # 说到一半改口
            (85, "帮我看看那个，等一下，我说错了，是想问依赖了哪些第三方库"),
            # 中英夹杂
            (112, "顺便说说这个项目用的 python 版本和 async 框架"),
            # 元请求
            (140, "你能说得简短一点吗，我听着有点累"),
            # 指代前文
            (165, "你刚才说的第二点，再展开讲讲"),
            # 数字和单位
            (190, "端到端延迟现在是多少毫秒，比原来快了百分之多少"),
            # 并发里选择性取消
            (215, "把查依赖那个任务取消掉，另一个继续"),
            (240, "现在还剩几个任务在跑"),
            # 只回一个字
            (265, "嗯"),
            # 情绪化催促
            (290, "怎么这么慢啊，还要等多久"),
            (320, "算了都停了吧"),
            # 停完再来新的
            (345, LATENCY),
            (375, "这次快点"),
            (405, "好了没"),
            (440, "把结果压缩成三句话讲给我"),
            (475, "刚才你提到的那个观察者，它是怎么挂上去的"),
            (510, "行，最后一个问题，这套东西还有什么明显短板"),
            (555, "好的谢谢"),
        ],
        total=600,
    ),
    Scenario(
        name="选择性取消",
        why=(
            "并发跑两个任务，只停其中一个。之前 cancel_task 是一刀切全停，"
            "用户说「把查依赖那个停掉，另一个继续」根本做不到"
        ),
        script=[
            (0, ARCH),
            (22, "另外再看看这个项目依赖了哪些第三方库"),
            (48, "把查依赖那个任务停掉，架构那个继续"),
            (70, "现在还剩几个在跑"),
        ],
        total=160,
    ),
    Scenario(
        name="并发调度",
        why=(
            "并发上限设成 1，连抛三个重任务。第二个来的时候名额已满，调度器要问 LLM "
            "怎么办——排队、抢占还是挂起。第三个明确说「这个最急」，看它会不会挂起"
            "在跑的那个让急的先上，之后再把挂起的恢复回来"
        ),
        script=[
            (0, HEAVY),
            (25, "另外再看看这个项目依赖了哪些第三方库"),
            (55, "先停一下，这个最急：voice_bot.py 里的填充语是怎么实现的"),
            (95, "现在几个任务在跑，分别什么状态"),
            (150, "都查完了吗"),
        ],
        total=260,
    ),
    Scenario(
        name="挂起恢复",
        why=(
            "专门构造 pause 该被选中的局面：先让一个重任务跑到读了好几个文件，"
            "再抛一个明确更急的小问题。调度器应该挂起而不是取消——取消会丢掉"
            "已经读的那些文件。急的答完之后，挂起的那个要能自己接着跑完"
        ),
        script=[
            (0, HEAVY),
            # 等它读进去一些文件再插队，这时取消的代价才明显
            (75, "打断一下，我现在就要知道，填充语那个文件有多少行"),
            (110, "刚才那个大任务还在吗，什么状态"),
            (200, "两个都好了吗"),
        ],
        total=320,
    ),
    Scenario(
        name="抢占风暴",
        why=(
            "并发设 1，每隔二十秒抛一个都声称最急的任务，连抛五个。看调度器会不会"
            "无脑抢占导致谁都跑不完（每个都被下一个干掉），还是能守住"
            "「跑了很久快出结果的别动」"
        ),
        # 台词一律用中文说法：本地 TTS 念英文标识符不准，听写会变成乱码，
        # 场景就测不到调度本身了（实测「最急：filler.py」被听成「最急富败」）。
        script=[
            (0, "最急，填充语那块是怎么工作的"),
            (20, "更急，过滤器那个文件里有哪几个过滤器"),
            (40, "这个最优先，调度器的暂停是怎么实现的"),
            (60, "先别管前面的，看看进度接口那块"),
            (80, "算了都不急了，就说说整体架构吧"),
            (140, "刚才那几个问题，哪些查出来了"),
        ],
        total=300,
    ),
    Scenario(
        name="改主意重派",
        why=(
            "完整生命周期：派任务 → 反悔取消 → 换个问题重派 → 问进度 → 拿结果。"
            "取消之后工具表要恢复，否则第二次派不出去"
        ),
        script=[
            (0, ARCH),
            (15, "等等，不查架构了，取消吧。"),
            (25, LATENCY),
            (55, "现在查到哪儿了？"),
            (90, "结果出来了吗？"),
        ],
        total=150,
    ),
]


async def main():
    args = sys.argv[1:]
    if "--list" in args:
        for s in SCENARIOS:
            print(f"{s.name:<12s} {s.total:>5.0f}s  {s.why}")
        return

    voice = "--text" not in args
    keyword = next((a for a in args if not a.startswith("--")), "")
    mode = "语音输入（合成台词→听写）" if voice else "文本注入（快模式）"
    for scenario in SCENARIOS:
        if keyword and keyword not in scenario.name:
            continue
        print(f"\n{'=' * 72}")
        print(f"场景：{scenario.name}（{scenario.total:.0f} 秒，{mode}）")
        print(f"看点：{scenario.why}")
        print("=" * 72)
        h = Harness(voice=voice)
        await h.start()
        try:
            await h.run(scenario)
        finally:
            await h.stop()
        h.transcript.dump()


if __name__ == "__main__":
    asyncio.run(main())
