"""判断这句话是不是在跟助手说——分层判定，默认只拦远场背景。

学术上这件事叫 device-directed speech detection（DDSD）。多人在场时，助手听到的
不全是对它说的：旁人闲聊、用户自言自语、电视背景音，照单全收就会乱插话。

**两个信号判的不是同一件事**，实测出来的（见 acoustic_probe.py / addressee_probe.py）：

* 音量判**距离**。近讲 RMS 0.13-0.19、中距 0.046-0.068、远处 0.016-0.024，三档完全
  分开。但同样近讲时「这个功能怎么实现」和「中午吃什么」音量一模一样——它对意图
  一无所知。
* 关键词判**意图指向**。14 句边界样本上准确率 93%，比想象中好。但它看不见距离。

所以分成两层，而不是加权揉进一个分数：先用音量滤掉远场背景，再用关键词判意图。
两层各管一维，谁失手都能从日志里看出是哪一层的问题。

第三层是 LLM 复核。它单用只有 79%（大量返回 0.5 弃权），但和关键词融合能到 100%
——两者错误模式互补。代价是 1.7 秒，**放不进语音主链路**，所以这里只把拿不准的句子
标记出来（``needs_review``），复核交给调用方在链路外做。

试过把三者揉进一个逻辑回归（addressee_train.py），留一交叉验证 78%，只比手调的
73% 高五个点，而且错误集中在「远场 + 对助手」——文本证据太强，音量权重压不住。
样本 60 条、特征 7 个手工维度，学不出真正的交互。那条路要走得像 Apple 论文那样上
音频 encoder 表征加真实数据，当前不划算。

    Pipeline([..., stt, AddresseeGate(), user_aggregator, ...])
"""

import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

LOG_PATH = Path(__file__).parent / "logs" / "addressee.jsonl"

# 远场阈值。实测远处 RMS 最大 0.0235、中距最小 0.0463，取中间。原来拍的 0.02 太低，
# 远处六句只拦下两句。
QUIET_RMS = float(os.getenv("ADDRESSEE_QUIET_RMS", "0.035"))
# 关键词分落在这个区间就是拿不准，交给链路外的 LLM 复核。
UNSURE_LOW = 0.30
UNSURE_HIGH = 0.52

# 冲着助手来的说法：祈使、请求、第二人称提问。
_TO_ASSISTANT = re.compile(
    r"帮我|帮忙|你能|你可以|你知道|告诉我|查一下|看一下|说说|讲讲|介绍一下|"
    r"打开|关掉|停一下|继续|重复|再说一遍|怎么办|是什么|为什么|多少|"
    # 这几条是实测漏掉的：「把结果讲简单点」「现在查到哪一步了」都是明确的指令，
    # 但原来的表一个都不命中。
    r"简单点|详细点|查到哪|到哪一步|好了没|完了没|结果呢|接着说|换一个"
)
# 明显是人跟人聊的：第三人称八卦、生活话题、招呼语。
_TO_HUMAN = re.compile(
    r"他说|她说|他们|昨天|前天|下班|吃饭|午饭|晚饭|咖啡|球赛|周末|老板|"
    r"你觉得呢|是吧|对吧|哈哈|哎呀|那个谁|回头聊|先走了"
)
# 项目术语出现，多半是在问助手。
_DOMAIN = re.compile(
    r"代码|文件|架构|延迟|测试|接口|模块|依赖|函数|项目|日志|性能|并发|"
    r"听写|合成|语音|模型"
)


@dataclass
class Utterance:
    """一句话的判定结果。"""

    text: str = ""
    secs: float = 0.0
    rms: float = 0.0
    score: float = 0.0
    label: str = ""
    needs_review: bool = False
    signals: dict = field(default_factory=dict)


def _rms(pcm: bytes) -> float:
    """算一段 16 位 PCM 的均方根音量，归一到 0-1。"""
    if len(pcm) < 2:
        return 0.0
    total = 0
    # 每隔若干个采样点取一个就够估音量了，没必要全算。
    step = max(2, (len(pcm) // 2 // 400) * 2)
    count = 0
    for i in range(0, len(pcm) - 1, step):
        v = int.from_bytes(pcm[i : i + 2], "little", signed=True)
        total += v * v
        count += 1
    return math.sqrt(total / count) / 32768.0 if count else 0.0


def keyword_score(text: str) -> tuple[float, dict]:
    """纯文本的意图指向分，不看声学。

    去掉了「有没有『你』」和「是不是疑问句」——逻辑回归给这两个学到的权重是
    -0.03 和 -0.02，基本等于噪声，留着只会让「中午吃什么？」这种因为带问号而虚高。

    Args:
        text: 听写出来的一句话。

    Returns:
        (0 到 1 的分数, 各信号命中次数)。
    """
    to_assistant = len(_TO_ASSISTANT.findall(text))
    to_human = len(_TO_HUMAN.findall(text))
    domain = len(_DOMAIN.findall(text))

    score = 0.35
    score += 0.20 * min(to_assistant, 2)
    score += 0.12 * min(domain, 2)
    score -= 0.22 * min(to_human, 2)
    return (
        max(0.0, min(1.0, score)),
        {"to_assistant": to_assistant, "to_human": to_human, "domain": domain},
    )


class AddresseeGate(FrameProcessor):
    """分层判断每句话是不是冲着助手来的。

    拦截分三档（``mode``），差别在于**信任哪一层**：

    * ``off``——只记录不拦。
    * ``volume``——只拦音量判出来的远场背景。默认档。
    * ``all``——凡是判定不等于 ``assistant`` 的一律拦掉。

    默认只信音量层，是因为两层的可靠度差着量级。噪声实测（``noise_probe.py``
    的 ``GATE=1``）：用户近讲 RMS 0.14-0.17，旁人在旁边聊 0.040，旁人在远处 0.010
    ——远场那档差十几倍，怎么切都不会误伤。而关键词层的基准分是 0.35、阈值 0.45，
    意味着**一句不含指令词也不含项目术语的话会被判成不是对助手说的**，「给我讲个
    笑话」这种正当请求就这么被拦了。误拦用户的代价远高于偶尔听进一句旁人的话，
    所以关键词那层只记不拦。

    **默认是 ``off``，因为音量阈值没在真实麦克风上标定过就是有害的。**合成音频的近讲
    RMS 是 0.13-0.19，据此定的远场阈值 0.035 看着很宽松；换成浏览器 WebRTC 麦克风实测，
    用户正常说话的 RMS 中位只有 0.031、范围 0.0035-0.068——阈值卡在了真实语音的中位数
    **之上**，11 句拦掉 8 句，包括「可可以可以，听到吗？」这种明显在对助手说的话。
    用户体感就是说了没反应、得重说一遍。

    标定方法：把 ``ADDRESSEE`` 留在 ``off`` 跑一段真实对话，从 ``logs/addressee.jsonl``
    读自己说话时的 RMS 分布，再让旁人在同样距离聊几句，看两组分不分得开。分得开才设
    ``ADDRESSEE_QUIET_RMS`` 并打开 ``volume``；分不开就别开——这台机器上就分不开，
    用户自己说话的音量从 0.0035 到 0.068 横跨了整个区间。
    """

    def __init__(
        self,
        *,
        mode: str = "volume",
        threshold: float = 0.45,
        quiet_rms: float = QUIET_RMS,
        log_path: Path | None = None,
        **kwargs,
    ):
        """初始化。

        Args:
            mode: ``off`` 只观测，``volume`` 只拦远场背景，``all`` 全拦。
            threshold: 关键词分低于这个算「不是对助手说的」。只影响记录里的
                ``label``，``volume`` 档下不据此拦截。
            quiet_rms: 音量低于这个直接当远场背景，不再看文本。
            log_path: 观测记录落盘位置。
            **kwargs: 透传给 FrameProcessor。
        """
        super().__init__(**kwargs)
        self._mode = mode
        self._threshold = threshold
        self._quiet = quiet_rms
        self._path = log_path or LOG_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._pcm = bytearray()
        self._rate = 16000
        # 内存里也留一份。测量脚本从这里读，比回头解析 jsonl 可靠——一句话被 VAD
        # 切成多段时会产生多条记录，按文件行数去对应输入句子会错位。
        self.records: list[Utterance] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """攒当前这句的音频，听写出来时判定。"""
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._pcm.clear()
        elif isinstance(frame, InputAudioRawFrame):
            # 只留够算音量的量，别把整段录音攒在内存里。
            if len(self._pcm) < 16000 * 2 * 30:
                self._pcm.extend(frame.audio)
                self._rate = frame.sample_rate
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            utt = self.judge(frame.text.strip())
            self._record(utt)
            if self._blocks(utt):
                logger.info(f"{self}: 判为{utt.label}，拦下「{utt.text}」")
                return

        await self.push_frame(frame, direction)

    def _blocks(self, utt: Utterance) -> bool:
        """这条要不要拦。"""
        if self._mode == "all":
            return utt.label != "assistant"
        if self._mode == "volume":
            return utt.label == "background"
        return False

    def judge(self, text: str) -> Utterance:
        """分层判定一句话。

        Args:
            text: 听写出来的一句话。

        Returns:
            判定结果。``label`` 是 ``assistant`` / ``other`` / ``background``。
        """
        secs = len(self._pcm) / (self._rate * 2) if self._pcm else 0.0
        rms = _rms(bytes(self._pcm))
        score, signals = keyword_score(text)

        # 第一层：音量。远场背景连听写都不可靠，不必再看文本说了什么。
        #
        # 条件用「有没有音频」而不是「rms 是不是真值」：写成 ``if rms and ...`` 时
        # rms 恰好为 0 会因为假值跳过这一层，落到关键词层判成 other——最安静的那种
        # 反而躲过了音量门，在只拦 background 的档位下直接放行。这个方向的失效必须堵住。
        # 真正没采到音频（``_pcm`` 空）另说，那时无从判起，只能交给关键词层。
        if self._pcm and rms < self._quiet:
            label, review = "background", False
        else:
            # 第二层：意图指向。
            label = "assistant" if score >= self._threshold else "other"
            # 第三层：拿不准的标出来，复核交给链路外——LLM 判一次要一秒七，
            # 语音这条路等不起。
            review = UNSURE_LOW <= score <= UNSURE_HIGH

        return Utterance(
            text=text,
            secs=round(secs, 2),
            rms=round(rms, 4),
            score=round(score, 3),
            label=label,
            needs_review=review,
            signals=signals,
        )

    def pending_review(self) -> list[Utterance]:
        """拿不准、值得用 LLM 复核的那些句子。"""
        return [u for u in self.records if u.needs_review]

    def _record(self, utt: Utterance) -> None:
        self.records.append(utt)
        mark = " ?" if utt.needs_review else ""
        logger.info(
            f"{self}: {utt.score:.2f} [{utt.label}{mark}] "
            f"rms={utt.rms:.3f} 「{utt.text[:30]}」"
        )
        with self._path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({"at": time.time(), **utt.__dict__}, ensure_ascii=False)
                + "\n"
            )


JUDGE_SYSTEM = (
    "判断用户这句话是在跟语音助手说，还是在跟旁边的人说话（助手只是碰巧听见）。\n"
    "跟助手说：提要求、问问题、下指令、要它停下或继续，通常带「帮我」「查一下」"
    "「你能」这类，或者在问技术/事务性的事。\n"
    "跟人说：聊天气吃饭八卦、讲第三个人的事、对旁人的附和与调侃、自言自语。\n"
    "只输出一个 0 到 1 的小数：1 表示确定在跟助手说，0 表示确定在跟人说，"
    "拿不准给 0.5。不要输出别的任何字。"
)


async def llm_judge(text: str, client, model: str, timeout: float = 12.0) -> float:
    """让模型判断这句话是不是对助手说的。

    关键词计数很脆——「这个功能是怎么实现的」没有指令词就被判低。模型看的是整句
    意图，边界样本上稳得多。但它自己也不可靠：14 句里有 7 句返回 0.5 弃权，单用
    准确率 79%，**低于关键词的 93%**。价值在于两者错误模式互补，融合能到 100%。

    代价是一次网络往返，所以**别放在语音主链路上**（那条路径的预算只有一秒多）。

    Args:
        text: 听写出来的一句话。
        client: AsyncOpenAI 兼容客户端。
        model: 模型名。
        timeout: 超时秒数，超了返回 0.5（不表态）。实测单次判断中位 1.7 秒、
            尾巴到 3.6 秒，所以别设太紧——4 秒时大部分调用都超时，
            准确率会假摔到 50%（全是不表态）。

    Returns:
        0 到 1 的分数。任何失败都返回 0.5，让调用方去跟关键词分融合。
    """
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": text},
                ],
                max_tokens=8,
                temperature=0,
            ),
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "").strip()
        m = re.search(r"[01](?:\.\d+)?", raw)
        return max(0.0, min(1.0, float(m.group(0)))) if m else 0.5
    except Exception as e:  # noqa: BLE001 - 判不出来就不表态，不能挡住对话
        # 要打类型：超时异常的 str() 是空的，只打 message 会看到一行空日志，
        # 完全看不出是超时还是别的。
        logger.warning(f"受话人判断失败，按不表态处理：{type(e).__name__} {e}")
        return 0.5


def fuse(keyword: float, llm: float) -> float:
    """把关键词分和 LLM 分合起来。

    **弃权不能当成中间意见。**LLM 返回 0.5 是「我不表态」，直接取平均会把关键词分
    往 0.5 拽——实测「把结果讲简单点」关键词 0.35、LLM 弃权，平均成 0.425 仍然低于
    阈值，等于白复核一场。所以弃权时原样退回关键词分。

    Args:
        keyword: 关键词分。
        llm: LLM 分，0.5 附近表示它没表态。

    Returns:
        融合分。
    """
    if 0.45 <= llm <= 0.55:
        return keyword
    return (keyword + llm) / 2


def build_gate() -> AddresseeGate:
    """按环境变量装配。``ADDRESSEE`` 取 ``off``（默认）/ ``volume`` / ``all``。

    默认不拦。阈值是绝对音量，而**真实麦克风和合成音频差着一个数量级**，没在本机标过
    就开等于随机丢用户的话。要开先按下面「标定」那节走一遍。
    """
    mode = os.getenv("ADDRESSEE", "off")
    # 兼容旧写法：以前 ADDRESSEE=enforce 表示「凡不是对助手说的都拦」。
    if mode == "enforce":
        mode = "all"
    return AddresseeGate(
        mode=mode,
        threshold=float(os.getenv("ADDRESSEE_THRESHOLD", "0.45")),
    )
