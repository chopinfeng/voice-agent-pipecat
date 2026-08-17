"""逐环节延迟统计：观察管线里的帧流动，按对话轮次记录时间线。

挂到 PipelineWorker 的 observers 上即可，不需要改动管线结构：

    worker = PipelineWorker(pipeline, observers=[LatencyObserver()])

每轮对话结束（机器人说完）时往日志打一张时间线表格，同时追加一行 JSON 到
``logs/latency.jsonl``，方便事后统计多轮的分布。
"""

import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    ProcessingMetricsData,
    TTFAMetricsData,
    TTFBMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

# 时间线上的观测点，按发生顺序排列。每个条目是 (标签, 帧类型, 只记第一次)。
# 顺序决定日志里的行序，所以这里就是这条管线的「全流程」定义。
TIMELINE: list[tuple[str, type[Frame], bool]] = [
    ("用户开始说话", VADUserStartedSpeakingFrame, True),
    ("VAD 判定说完", VADUserStoppedSpeakingFrame, True),
    ("首个临时听写", InterimTranscriptionFrame, True),
    ("听写结果返回", TranscriptionFrame, True),
    ("轮次结束放行", UserStoppedSpeakingFrame, True),
    ("LLM 请求发出", LLMFullResponseStartFrame, True),
    ("LLM 首 token", LLMTextFrame, True),
    ("TTS 开始合成", TTSStartedFrame, True),
    ("第一个音频帧", TTSAudioRawFrame, True),
    ("机器人开始出声", BotStartedSpeakingFrame, True),
    ("LLM 输出完毕", LLMFullResponseEndFrame, False),
    ("TTS 合成完毕", TTSStoppedFrame, False),
    ("机器人说完", BotStoppedSpeakingFrame, False),
]

# 单独拎出来汇报的关键区间，(标签, 起点, 终点)。「端到端」量的是用户说完之后多久
# 听到声音——开了填充语时第一声就是填充语，真实回答什么时候接上看「回答开始合成」。
# 听写和轮次判定是并行跑的——
# 停顿窗口从 VAD 说完就开始计时，不是等听写回来才开始——所以两个都从「VAD 判定
# 说完」量起，谁慢谁决定什么时候能进 LLM。
SPANS: list[tuple[str, str, str]] = [
    ("听写", "VAD 判定说完", "听写结果返回"),
    ("轮次判定", "VAD 判定说完", "轮次结束放行"),
    ("LLM 首 token", "LLM 请求发出", "LLM 首 token"),
    ("回答开始合成", "VAD 判定说完", "TTS 开始合成"),
    ("端到端 说完→出声", "VAD 判定说完", "第一个音频帧"),
]


@dataclass
class Turn:
    """一轮对话里各观测点的时间戳（纳秒，管线时钟）。"""

    index: int
    marks: dict[str, int] = field(default_factory=dict)
    ttfb: dict[str, float] = field(default_factory=dict)
    ttfa: dict[str, float] = field(default_factory=dict)
    processing: dict[str, float] = field(default_factory=dict)
    transcript: str = ""
    reply: str = ""


class LatencyObserver(BaseObserver):
    """按轮次统计管线各环节耗时，输出到日志和 JSONL。"""

    def __init__(
        self,
        *,
        log_path: Path | None = None,
        finish_on: tuple[type[Frame], ...] = (BotStoppedSpeakingFrame,),
        **kwargs,
    ):
        """初始化。

        Args:
            log_path: JSONL 输出路径，默认 ``logs/latency.jsonl``。传别的路径可以
                把不同实验的数据分开。
            finish_on: 哪些帧算一轮结束。默认是输出 transport 发的
                ``BotStoppedSpeakingFrame``。离线回放没有 transport，传
                ``(TTSStoppedFrame,)``——代价是多句回答会被拆成多轮。
            **kwargs: 透传给 BaseObserver。
        """
        super().__init__(**kwargs)
        self._finish_on = finish_on
        self._path = log_path or Path(__file__).parent / "logs" / "latency.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._turn = Turn(index=0)
        self._count = 0
        # 同一个帧会在每一对相邻处理器之间各推一次，只认第一次。去重窗口跨轮次，
        # 否则一个帧在传播途中开了新轮，后面的处理器又会把它当成新帧。
        self._seen: deque[int] = deque(maxlen=8192)
        self._seen_set: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        """记录每个观测点第一次出现的时刻，并在轮次结束时汇报。

        统计出错绝不能影响它观察的管线，所以整个方法兜住异常：一个抛出去的
        observer 会被上游当成推帧失败，后面的帧就再也送不到这里了。
        """
        try:
            await self._observe(data)
        except Exception as e:  # noqa: BLE001 - 观测失败不该影响管线
            logger.warning(f"延迟统计失败（已忽略）: {type(data.frame).__name__}: {e}")

    async def _observe(self, data: FramePushed):
        frame = data.frame

        if frame.id in self._seen_set:
            return
        if len(self._seen) == self._seen.maxlen:
            self._seen_set.discard(self._seen[0])
        self._seen.append(frame.id)
        self._seen_set.add(frame.id)

        # 用户说话可能被 VAD 切成好几段，所以只有在这一轮已经开始回答之后再听到
        # 「开始说话」才算打断、才开新轮。
        if (
            isinstance(frame, VADUserStartedSpeakingFrame)
            and "LLM 请求发出" in self._turn.marks
        ):
            self.flush("被打断")

        turn = self._turn
        for label, kind, first_only in TIMELINE:
            if not isinstance(frame, kind):
                continue
            if first_only and label in turn.marks:
                continue
            turn.marks[label] = data.timestamp

        if isinstance(frame, TranscriptionFrame):
            turn.transcript += frame.text
        elif isinstance(frame, LLMTextFrame):
            turn.reply += frame.text
        elif isinstance(frame, MetricsFrame):
            self._collect_metrics(turn, frame)

        # 填充语会先于正式回答播完一次，那一下的收尾帧不能算轮次结束——等 LLM
        # 这一轮的输出真的收尾了再汇报。
        if isinstance(frame, self._finish_on) and "LLM 输出完毕" in turn.marks:
            self.flush("完成")

    def _collect_metrics(self, turn: Turn, frame: MetricsFrame):
        """收下各服务自己上报的 TTFB / TTFA / 处理耗时，作为时间线的交叉验证。

        只取 metrics 里的数值，不拿这个帧的到达时刻当观测点：``MetricsFrame`` 是
        SystemFrame，会插到队列前面，实测比同时推出的 ``LLMFullResponseStartFrame``
        还早到，当时间戳用会得出负数区间。
        """
        for item in frame.data:
            if isinstance(item, TTFBMetricsData):
                turn.ttfb[item.processor] = item.value
            elif isinstance(item, TTFAMetricsData):
                # TTFA 拆成了服务响应时间和前导静音两部分，记总时长。
                turn.ttfa[item.processor] = item.ttfa
            elif isinstance(item, ProcessingMetricsData):
                turn.processing[item.processor] = (
                    turn.processing.get(item.processor, 0.0) + item.value
                )

    def flush(self, reason: str = "手动"):
        """汇报当前轮次并开启下一轮。

        真实 bot 里由 ``BotStoppedSpeakingFrame`` 自动触发。离线回放没有输出
        transport，也就没有那个帧，需要在跑完之后手动调一次。

        Args:
            reason: 记进日志的收尾原因。
        """
        turn = self._turn
        self._turn = Turn(index=turn.index + 1)
        # 结束帧会成对广播（上下游各一个，id 不同），第二个会开出一轮只有收尾标记
        # 的空轮次。没走到 LLM 的轮次没有统计价值，直接丢掉。
        if "LLM 请求发出" not in turn.marks:
            return
        self._count += 1

        # 按实际发生时刻排序，而不是按 TIMELINE 的定义顺序——比如 LLM 输出完毕
        # 常常早于 TTS 开始合成，照定义顺序打出来会显得时间倒流。
        ordered = sorted(turn.marks.items(), key=lambda kv: kv[1])
        origin = ordered[0][1]

        lines = [f"轮次 #{self._count} ({reason})"]
        if turn.transcript:
            lines.append(f"  听到 : {turn.transcript.strip()}")
        if turn.reply:
            lines.append(f"  回答 : {turn.reply.strip()}")
        lines.append(f"  {'环节':<16s}{'距上一步':>10s}{'累计':>10s}")
        prev = origin
        for label, ts in ordered:
            lines.append(
                f"  {label:<16s}{(ts - prev) / 1e9:9.2f}s{(ts - origin) / 1e9:9.2f}s"
            )
            prev = ts

        spans = {}
        for label, a, b in SPANS:
            if a in turn.marks and b in turn.marks:
                spans[label] = (turn.marks[b] - turn.marks[a]) / 1e9
        if spans:
            lines.append("  " + "  ".join(f"{k} {v:.2f}s" for k, v in spans.items()))
        for name, values in (("TTFB", turn.ttfb), ("TTFA", turn.ttfa)):
            if values:
                lines.append(
                    f"  服务自报 {name}: "
                    + "  ".join(f"{k} {v:.2f}s" for k, v in sorted(values.items()))
                )
        logger.info("\n".join(lines))

        record = {
            "turn": self._count,
            "reason": reason,
            "transcript": turn.transcript.strip(),
            "reply": turn.reply.strip(),
            "timeline_s": {label: (ts - origin) / 1e9 for label, ts in ordered},
            "spans_s": spans,
            "ttfb_s": turn.ttfb,
            "ttfa_s": turn.ttfa,
            "processing_s": turn.processing,
        }
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
