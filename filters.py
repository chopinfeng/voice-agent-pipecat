"""管线过滤器：清理会干扰下游的内容。

听写清洗那几个放在 STT 之后、上下文聚合器之前：

    Pipeline([..., stt, StripEmotionMarks(), TermCorrection(),
              EmptyTranscriptionFilter(), user_agg, ...])

``CollapseUserTurns`` 放在聚合器之后、LLM 之前。
"""

import os
import re
import time

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMTextFrame,
    TranscriptionFrame,
)
from pipecat.processors.aggregators.llm_response_universal import LLMAssistantAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# SenseVoice 会在转写里塞情感和事件标记（😊😔🎼 之类）。它们不是用户说的话，
# 进了上下文会被模型当成语气提示，还会被 TTS 当成文本念出来。
_EMOTION = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\uFE0F]|<\|[^|]*\|>"
)

# 只剩标点、空白或语气符号的转写视为空。
_MEANINGLESS = re.compile(r"^[\s。，、．,\.！？!?…~～·、;；:：\-—_\"'“”‘’()（）]*$")


# 中文语音里夹的英文技术词，本地 TTS 念不准、SenseVoice 又听岔，就成了固定的错法。
# 只做整词替换，且表保持短——这是权宜之计，不是通用纠错器。SenseVoice 不支持热词
# （那是 Paraformer 的能力），pipecat 也没暴露相应接口，所以只能在后面兜。
TERM_FIXES = {
    "back": "python",
    "better": "python",
    "拍摄": "python",
    "拍神": "python",
    "派森": "python",
    "赛顿": "python",
}
# 边界要连连字符一起挡掉，否则 "back-end" 会被改成 "python-end"。
_TERM_RE = re.compile(
    "|".join(rf"(?<![A-Za-z-]){re.escape(k)}(?![A-Za-z-])" for k in TERM_FIXES),
    re.IGNORECASE,
)


class TermCorrection(FrameProcessor):
    """把听岔的技术术语纠回来。

    放在 STT 之后：本地 TTS 念 "python" 发音不准，听写会稳定地错成 "back" 或
    "better"，下游 agent 就真去搜名字带 back 的文件了。
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """就地纠正 TranscriptionFrame 的文本，其余原样放行。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            fixed = _TERM_RE.sub(
                lambda m: TERM_FIXES[m.group(0).lower()], frame.text
            )
            if fixed != frame.text:
                logger.debug(f"{self}: 术语纠正 {frame.text!r} -> {fixed!r}")
                frame.text = fixed

        await self.push_frame(frame, direction)


class StripEmotionMarks(FrameProcessor):
    """去掉听写结果里的情感/事件标记。

    放在 ``EmptyTranscriptionFilter`` 之前——先把标记清掉，只剩标点的那种才认得出
    是空转写。
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """就地清洗 TranscriptionFrame 的文本，其余原样放行。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and frame.text:
            cleaned = _EMOTION.sub("", frame.text).strip()
            if cleaned != frame.text:
                logger.debug(f"{self}: 清掉情感标记 {frame.text!r} -> {cleaned!r}")
                frame.text = cleaned

        await self.push_frame(frame, direction)


class EmptyTranscriptionFilter(FrameProcessor):
    """丢掉没有实际内容的听写结果。

    用户说完之后的那段静音会被识别成一个孤零零的句号，聚合器把它当成新一轮用户
    发言，于是每轮都白打一次 LLM——多花一次钱，还会让助手对着标点再答一句。
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """拦下空转写，其余原样放行。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame) and _MEANINGLESS.match(
            frame.text or ""
        ):
            logger.debug(f"{self}: 丢弃空转写 {frame.text!r}")
            return

        await self.push_frame(frame, direction)


# 合并标注的开头。剥旧标注时按它来找。
_COLLAPSE_MARK = "[系统提示，不要向用户复述或提及本段]"
# 合并后的正文只保留「只回应这一句」后面那部分。
_COLLAPSE_TAIL = "只回应这一句——\n"


def _strip_mark(content: str) -> str:
    """把上一轮合并留下的标注剥掉，只留真正的用户原话。

    按**最后**一个分隔符切，不是第一个：嵌套时内层标注整个躺在外层的「已作废」
    那一段里，第一个分隔符属于内层，照它切只会剥掉一半。
    """
    content = content.strip()
    while content.startswith(_COLLAPSE_MARK) and _COLLAPSE_TAIL in content:
        content = content.rsplit(_COLLAPSE_TAIL, 1)[1].strip()
    return content


# 两句之间超过这么久就不算改口，各自都要处理。
COLLAPSE_WINDOW = float(os.getenv("COLLAPSE_WINDOW", "6"))


class CollapseUserTurns(FrameProcessor):
    """把上下文尾部连着的几条用户发言并成一条。

    用户打断助手时，上一轮回答被取消，但那条用户消息已经进了上下文。连打断几次，
    上下文尾部就攒下三四条谁也没回答过的用户发言，模型会把它们当成并列的几个请求
    一起消化——实测连续打断三次后要 52 秒才开口，答出来的东西还串味。

    真实意图是后面的覆盖前面的（「说说项目」→「等等先告诉我几点」→「行你继续」）。
    所以合并成一条，把先前那几句标成背景，只让最后一句作为要回应的请求。

    **只合并短时间内连着说的。**隔了好几秒才说的下一句通常是另一件事，不是改口——
    「把项目看一遍」隔十秒再说「先别管那个，今天天气怎么样」，两句都得处理，合并会
    把前面那个查询请求整个吞掉（实测就这么丢过一次派单）。窗口默认六秒。

    只在送进 LLM 的那一刻改写，不动聚合器维护的原始上下文——历史该留还是留着。
    """

    def __init__(self, **kwargs):
        """初始化。"""
        super().__init__(**kwargs)
        self._last_seen = 0.0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """合并尾部连续的用户发言，其余原样放行。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMContextFrame):
            now = time.monotonic()
            recent = (now - self._last_seen) <= COLLAPSE_WINDOW
            self._last_seen = now
            if recent:
                self._collapse(frame.context)

        await self.push_frame(frame, direction)

    def _collapse(self, context) -> None:
        messages = list(context.get_messages())
        tail = []
        while messages and messages[-1].get("role") == "user":
            tail.append(messages.pop())
        if len(tail) < 2:
            return

        tail.reverse()
        # 合并结果自己也是一条 user 消息，下一轮还会被合进来——不剥掉旧标注就会
        # 套娃（实测嵌到两层：「[系统提示…作废…[系统提示…作废…」）。用户连着插话时
        # 这一段每轮都触发，越滚越长，最后模型看到的几乎全是标注。
        latest = _strip_mark(str(tail[-1].get("content", "")))
        earlier = "；".join(_strip_mark(str(m.get("content", ""))) for m in tail[:-1])
        # 两条约束都得写死。一是"已作废"——只写"以这句为准"的话模型仍会顺手去处理
        # 前面那几句，实测会先查一遍根本不存在的后台任务白花十秒。二是"别提这段"——
        # 不写的话模型会把标注当内容复述出来，用户听到一句"你刚才说的那些已经作废了"
        # 完全摸不着头脑（实测在十分钟场景里冒出来过）。
        merged = (
            f"[系统提示，不要向用户复述或提及本段]用户话说到一半改了口，"
            f"下面这些已作废、不用回应：{earlier}。只回应这一句——\n{latest}"
        )
        logger.debug(f"{self}: 合并 {len(tail)} 条连续用户发言")
        context.set_messages(messages + [{"role": "user", "content": merged}])


# 模型判定「这句不是对我说的」时输出的标记。多人在场时它需要一个「选择不回应」的
# 出口——语音链路里模型总要吐点什么，所以约定一个标记，在进 TTS 之前拦掉。
SKIP_MARK = "[skip]"


class SkipFilter(FrameProcessor):
    """模型选择不回应时，别让标记被念出来。

    放在 LLM 和 TTS 之间。只吞文本，不动其他帧——轮次的开始结束信号照常传下去，
    否则聚合器会以为这一轮没结束。
    """

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """拦下只包含跳过标记的文本。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMTextFrame) and SKIP_MARK in frame.text:
            rest = frame.text.replace(SKIP_MARK, "").strip()
            if not rest:
                logger.debug(f"{self}: 模型判定不必回应，不出声")
                return
            # 标记混在正文里：把标记去掉，正文照念。
            frame.text = rest

        await self.push_frame(frame, direction)


class RememberInterrupted(LLMAssistantAggregator):
    """被打断时，把已经说出口的那半句也记进上下文。

    pipecat 默认的做法是 ``_handle_interruptions`` 直接 ``reset()``——**被打断的回复
    整个丢掉**。单次打断没什么，但用户连着插话时会滚成一个恶性循环：助手的话全都
    没进上下文，模型不记得自己刚说过什么，下一轮拿着几乎相同的上下文给出同一个
    答案，用户听到重复又插话。实测一段对话末尾连着四条用户发言中间一条助手消息
    都没有，模型每轮重复同一句。

    记下来的是**已经生成的那部分**，跟用户实际听到的接近（未必完全相同——TTS 可能
    还没念完就被切掉了）。宁可多记一点：模型知道自己已经讲过这个话题，比完全失忆
    更接近真相。
    """

    async def _handle_interruptions(self, frame):
        """先把说了一半的话提交进上下文，再走默认的收尾。

        ``_aggregation`` 是**分片的列表**不是字符串，取文本要走 ``aggregation_string()``
        ——直接 ``.strip()`` 会抛 AttributeError，而 FrameProcessor 会把异常吞成一条
        error 日志，表面上什么都没发生，实际上每次打断都没生效。
        """
        if self._aggregation and self.aggregation_string().strip():
            logger.debug(f"{self}: 被打断，先记下已说出口的部分")
            await self.push_aggregation()
        await super()._handle_interruptions(frame)
