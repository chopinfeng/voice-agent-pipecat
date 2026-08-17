"""区分说话人：给每句话标上是谁说的，交给模型自己决定回应谁。

跟 ``speaker_id.py`` 那套「只认注册者、其余一律拦掉」是两种思路。拦截把决策权交给
一个阈值，而阈值必须在「宁可漏进旁人」和「宁可吃掉用户的话」之间选边站——上一版
音量门选错了方向，11 句拦掉 8 句。这里改成**只标注不拦截**：每句话前面加一个
``[说话人N]``，让模型看着标签自己判断该答谁。判错的代价从「用户的话被吞掉」降到
「标签错了一次」，低了一个量级，阈值也就不再是生死线。

怎么分配编号：在线聚类。每句话算一个 192 维声纹向量，跟已知的几个说话人质心比余弦
相似度，最像的那个如果超过 ``JOIN``，就归给它并更新质心；都不够像就新开一个人。

``speaker_profile.npz`` 存在时，它当作**主用户**的种子，编号固定是 1；这样模型能分清
「常跟我说话的那个人」和「今天刚出现的旁人」。没有档案也能跑，只是所有人都按出现
顺序编号，谁是主人得模型自己从对话里推断。

短片段不判（``MIN_SECS``）。实测 0.3 秒时同一个人的相似度只有 0.42，硬分会把「嗯」
这种应答分给陌生人。短句一律标成上一个说话人——连续说话时这个猜测几乎总是对的。

判定过的音频都存进 ``testdata/speaker/``，配合 ``logs/diarize.jsonl`` 里的编号和分数，
以后调聚类阈值可以直接离线重跑。
"""

import asyncio
import json
import os
import time
import wave
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from speaker_id import MODEL_NAME, MODEL_RATE, PROFILE_PATH, Profile, cosine

HERE = Path(__file__).parent
LOG_PATH = HERE / "logs" / "diarize.jsonl"
CORPUS_DIR = HERE / "testdata" / "speaker"
SAVE_AUDIO = os.getenv("SPEAKER_SAVE_AUDIO", "1") != "0"

# 短于这个不算向量，直接归给上一个说话人。
MIN_SECS = float(os.getenv("DIARIZE_MIN_SECS", "1.2"))
# 相似度超过它就认为是同一个人。真实数据里同一个人跟自己质心的下限是 0.576，
# 合成的另一个人是 0.34，取中间偏低——**标错的代价只是一个标签**，
# 所以这里宁可偏向「算作同一个人」，免得同一个人被拆成好几个编号。
JOIN = float(os.getenv("DIARIZE_JOIN", "0.45"))
# 最多认几个人。超出的一律归到最后一个，避免嘈杂环境里编号无限膨胀。
MAX_SPEAKERS = int(os.getenv("DIARIZE_MAX", "6"))


class Speakers:
    """在线维护若干说话人的质心。"""

    def __init__(self, seed: np.ndarray | None = None):
        """初始化。

        Args:
            seed: 主用户的声纹质心，有就固定为 1 号。
        """
        self.centroids: list[np.ndarray] = []
        self.counts: list[int] = []
        self.seeded = seed is not None
        if seed is not None:
            self.centroids.append(seed)
            self.counts.append(1)

    def assign(self, vec: np.ndarray) -> tuple[int, float]:
        """把一条向量归给某个说话人。

        Returns:
            (说话人编号从 1 开始, 跟该说话人的相似度)。
        """
        if not self.centroids:
            self.centroids.append(vec)
            self.counts.append(1)
            return 1, 1.0

        scores = [cosine(vec, c) for c in self.centroids]
        best = int(np.argmax(scores))
        if scores[best] >= JOIN or len(self.centroids) >= MAX_SPEAKERS:
            # 滑动平均更新质心。主用户那条是注册出来的，也让它跟着适应当前环境
            # （麦克风、房间都会影响向量），但权重压得很低。
            n = self.counts[best]
            weight = 1.0 / min(n + 1, 20)
            self.centroids[best] = (1 - weight) * self.centroids[best] + weight * vec
            self.counts[best] = n + 1
            return best + 1, scores[best]

        self.centroids.append(vec)
        self.counts.append(1)
        return len(self.centroids), scores[best]

    def label(self, index: int) -> str:
        """编号转成给模型看的名字。"""
        if self.seeded and index == 1:
            return "主用户"
        return f"说话人{index}"


class Diarizer(FrameProcessor):
    """给每条听写结果标上说话人，不拦截。

    标注直接写进听写文本的前缀（``[主用户] 你好``），因为聚合器只往上下文里放文本
    ——想让模型看见说话人，就得写在文本里。系统提示里要相应地写明这个前缀是系统
    标注、不许念出来，否则模型会把它当内容复述（这个项目里已经栽过两次）。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        profile_path: Path = PROFILE_PATH,
        log_path: Path = LOG_PATH,
        **kwargs,
    ):
        """初始化。

        Args:
            enabled: 关掉就完全不碰帧。
            profile_path: 主用户声纹档案，没有也能跑。
            log_path: 每次判定的记录。
            **kwargs: 透传给 FrameProcessor。
        """
        super().__init__(**kwargs)
        self._enabled = enabled
        profile = Profile(profile_path)
        self._speakers = Speakers(profile.centroid if len(profile) else None)
        self._log = log_path
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._pcm = bytearray()
        self._rate = MODEL_RATE
        self._last = 1

    async def _ensure_model(self):
        if self._model is None:
            from funasr import AutoModel

            self._model = await asyncio.to_thread(
                AutoModel, model=MODEL_NAME, disable_update=True
            )
            seeded = "有主用户档案" if self._speakers.seeded else "无档案，按出现顺序编号"
            logger.debug(f"{self}: 声纹模型就绪（{seeded}）")

    async def _embed(self, pcm: bytes, rate: int) -> np.ndarray:
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != MODEL_RATE:
            idx = np.linspace(0, len(x) - 1, int(len(x) * MODEL_RATE / rate))
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        result = await asyncio.to_thread(lambda: self._model.generate(input=x))
        return np.asarray(result[0]["spk_embedding"]).ravel()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """攒音频，听写出来时给它加上说话人前缀。"""
        await super().process_frame(frame, direction)

        if not self._enabled:
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._pcm.clear()
        elif isinstance(frame, InputAudioRawFrame):
            if len(self._pcm) < MODEL_RATE * 2 * 30:
                self._pcm.extend(frame.audio)
                self._rate = frame.sample_rate
        elif isinstance(frame, TranscriptionFrame) and frame.text.strip():
            frame.text = await self._tag(frame.text.strip())

        await self.push_frame(frame, direction)

    async def _tag(self, text: str) -> str:
        """算出说话人，返回带前缀的文本。"""
        secs = len(self._pcm) / (self._rate * 2)
        if secs < MIN_SECS:
            # 太短，向量不可靠。归给上一个说话人——连着说话时这个猜测几乎总是对的。
            idx, score = self._last, None
        else:
            await self._ensure_model()
            vec = await self._embed(bytes(self._pcm), self._rate)
            idx, score = self._speakers.assign(vec)
            self._last = idx

        label = self._speakers.label(idx)
        self._record(text, secs, idx, score)
        logger.info(
            f"{self}: [{label}]"
            f"{f' {score:.2f}' if score is not None else ' 短句沿用'}"
            f"「{text[:24]}」"
        )
        return f"[{label}] {text}"

    def _record(
        self, text: str, secs: float, index: int, score: float | None
    ) -> None:
        clip = self._save_audio(index) if SAVE_AUDIO else None
        with self._log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "at": time.time(),
                        "speaker": index,
                        "score": round(score, 4) if score is not None else None,
                        "secs": round(secs, 2),
                        "text": text,
                        "clip": clip,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _save_audio(self, index: int) -> str | None:
        """原样存音频，文件名带说话人编号，供以后离线重标。"""
        if not self._pcm:
            return None
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        path = CORPUS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-spk{index}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._rate)
            w.writeframes(bytes(self._pcm))
        return str(path.relative_to(HERE))


def build_diarizer() -> Diarizer:
    """按环境变量装配。``DIARIZE=0`` 关掉。"""
    return Diarizer(enabled=os.getenv("DIARIZE", "1") != "0")
