"""声纹识别：只回应注册过的那个人，旁人说话直接丢掉。

这是 ``addressee.py`` 那套音量门失败之后的替代方案。音量判的是**距离**，而同事就坐在
旁边、音量跟用户差不多——实测真实麦克风下用户自己说话的 RMS 从 0.0035 到 0.068 横跨
整个区间，旁人必然落在里面，音量层原理上就分不开。声纹判的是**音色**，跟距离无关。

用 FunASR 的 CAM++（``speech_campplus_sv_zh-cn_16k-common``），输出 192 维向量，
比余弦相似度。它**不受音量影响**——同一段音频缩到真实麦克风的电平（RMS 0.0035-0.068）
相似度仍是 0.96-1.00，而音量门在同样的电平上完全失效。这是选它的根本原因。

**合成音频给的分离度是假的。**变调模拟的「另一个人」只有 0.27-0.34，看着中间空很宽；
换成真人短句，同一个人的两句话（「hello。」和「这声音太差了。」）实测只有 **0.43**。
所以：

* 短片段一律不判（``MIN_SECS``，默认 1.5 秒）。宁可漏进来一句，不能吃掉用户自己的话
  ——上一版音量门就是死在这个方向上的（11 句拦 8 句，用户说了没反应）。
* 注册片段要求更长（``ENROLL_MIN_SECS``），且跟已有样本差太远的不收。实测有过一条
  1.9 秒的「我。」，大半是静音，跟本人其余样本只有 0.05-0.28，混进质心就把档案带偏。
* **默认阈值 0.55 没有真实依据**，别直接开 ``verify``。先用 ``observe`` 跑一段，
  让旁人也说几句，看两组分数分不分得开。

每段判定过的音频都原样存进 ``testdata/speaker/``（``SPEAKER_SAVE_AUDIO=0`` 关掉），
配合 ``logs/speaker.jsonl`` 里的分数和文本，以后换模型或重标阈值可以直接离线重跑，
不用再请人对着麦克风重录一遍。

用法：

1. 注册（说五六句完整的话，每句两秒以上，**别让旁人插话**）::

       SPEAKER=enroll uv run --project pipecat python voice_bot.py -t webrtc

2. 观测——照常对话，让旁人也说几句，只打分不拦::

       SPEAKER=observe uv run --project pipecat python voice_bot.py -t webrtc

3. 看两组分数分不分得开，定阈值::

       uv run --project pipecat python speaker_id.py --stats

4. 分得开再开拦截::

       SPEAKER=verify SPEAKER_THRESHOLD=<标出来的值> uv run --project pipecat python voice_bot.py -t webrtc
"""

import asyncio
import json
import os
import sys
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

HERE = Path(__file__).parent
PROFILE_PATH = HERE / "speaker_profile.npz"
LOG_PATH = HERE / "logs" / "speaker.jsonl"
# 判定过的音频原样存这儿，供以后离线重标阈值用。
#
# 这个项目里反复出现同一种翻车：拿合成音频定阈值，接上真实麦克风就废——音量门的
# 0.035 卡在真实语音中位数之上（11 句拦 8 句），声纹的 0.55 也是合成数据给的假信心
# （真实短句同人只有 0.43）。留住真实音频才能一次标准、以后复用。
CORPUS_DIR = HERE / "testdata" / "speaker"
SAVE_AUDIO = os.getenv("SPEAKER_SAVE_AUDIO", "1") != "0"
MODEL_NAME = "iic/speech_campplus_sv_zh-cn_16k-common"
MODEL_RATE = 16000

# 低于这个时长不判，直接放行。真实数据比合成的散得多——同一个人的两句短话
# （「hello。」和「这声音太差了。」）实测只有 0.43，所以门槛比最初拍的 0.6 秒高不少。
MIN_SECS = float(os.getenv("SPEAKER_MIN_SECS", "1.5"))
THRESHOLD = float(os.getenv("SPEAKER_THRESHOLD", "0.55"))
# 注册用的片段要更长。1.9 秒的「我。」里大半是静音，算出来的向量跟本人其余样本
# 只有 0.05-0.28，混进质心就把整个档案带偏了。
ENROLL_MIN_SECS = float(os.getenv("SPEAKER_ENROLL_MIN_SECS", "2.0"))
# 跟已有样本差太远的一律不收——注册时旁人插一句，或者一段几乎全是静音的音频，
# 都会从这里被挡掉。
ENROLL_MIN_AGREE = float(os.getenv("SPEAKER_ENROLL_AGREE", "0.40"))
# 注册要几条。太少了质心不稳，太多了注册过程太累。
ENROLL_TARGET = int(os.getenv("SPEAKER_ENROLL", "5"))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """两个向量的余弦相似度。"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class Profile:
    """注册过的声纹：若干条向量，比对时用它们的质心。

    存多条而不是只存一个平均值，是为了以后能重算、能看单条——注册时如果混进一句
    别人的话，光看质心是发现不了的。
    """

    def __init__(self, path: Path = PROFILE_PATH):
        self.path = path
        self.vectors: list[np.ndarray] = []
        if path.exists():
            data = np.load(path)
            self.vectors = [data[k] for k in sorted(data.files)]

    def __len__(self) -> int:
        return len(self.vectors)

    @property
    def centroid(self) -> np.ndarray | None:
        """稳健质心：先把跟大多数不合群的样本剔掉再平均。

        单纯取平均是脆的——注册时混进一条噪声或旁人的话，质心就被拉偏，之后连本人
        都认不出。实测有过一条 1.9 秒的「我。」（大半是静音），跟其余样本相似度只有
        0.05-0.28，却照样进了平均。
        """
        if not self.vectors:
            return None
        if len(self.vectors) < 3:
            return np.mean(self.vectors, axis=0)
        keep = [v for v in self.vectors if self._agreement(v) >= ENROLL_MIN_AGREE]
        return np.mean(keep or self.vectors, axis=0)

    def _agreement(self, vec: np.ndarray) -> float:
        """这条跟**其余**样本的质心有多像。用于挑离群。"""
        rest = [v for v in self.vectors if v is not vec]
        return cosine(vec, np.mean(rest, axis=0)) if rest else 1.0

    def outliers(self) -> list[int]:
        """不合群的样本下标。"""
        return [
            i
            for i, v in enumerate(self.vectors)
            if len(self.vectors) >= 3 and self._agreement(v) < ENROLL_MIN_AGREE
        ]

    def add(self, vec: np.ndarray) -> bool:
        """加一条并落盘。

        Returns:
            True 表示收下了；False 表示跟已有样本差太远，判为不是同一个人（或者
            那段音频本身是噪声），没有收。
        """
        if len(self.vectors) >= 2 and cosine(vec, self.centroid) < ENROLL_MIN_AGREE:
            return False
        self.vectors.append(vec)
        np.savez(self.path, **{f"v{i}": v for i, v in enumerate(self.vectors)})
        return True

    def score(self, vec: np.ndarray) -> float:
        """跟质心比的相似度。"""
        c = self.centroid
        return cosine(vec, c) if c is not None else 0.0


class SpeakerGate(FrameProcessor):
    """按声纹决定这句话要不要往下传。

    位置跟 ``AddresseeGate`` 一样，挂在听写之后、聚合器之前：要拿到整段音频算向量，
    也要能在听写结果进上下文之前把它拦掉。
    """

    def __init__(
        self,
        *,
        mode: str = "off",
        threshold: float = THRESHOLD,
        min_secs: float = MIN_SECS,
        profile_path: Path = PROFILE_PATH,
        log_path: Path = LOG_PATH,
        **kwargs,
    ):
        """初始化。

        Args:
            mode: ``off`` 不做任何事，``enroll`` 收集声纹，``observe`` 只打分不拦，
                ``verify`` 拦非注册者。
            threshold: 相似度低于这个算不是注册者。
            min_secs: 短于这个不判，直接放行。
            profile_path: 声纹存档。
            log_path: 每次判定的记录，用来事后标阈值。
            **kwargs: 透传给 FrameProcessor。
        """
        super().__init__(**kwargs)
        self._mode = mode
        self._threshold = threshold
        self._min_secs = min_secs
        self._profile = Profile(profile_path)
        self._log = log_path
        self._log.parent.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._pcm = bytearray()
        self._rate = MODEL_RATE
        self._warned = False

    async def _ensure_model(self):
        """第一次用到时才加载，别拖慢启动。"""
        if self._model is None:
            from funasr import AutoModel

            self._model = await asyncio.to_thread(
                AutoModel, model=MODEL_NAME, disable_update=True
            )
            logger.debug(f"{self}: 声纹模型就绪（已注册 {len(self._profile)} 条）")

    async def _embed(self, pcm: bytes, rate: int) -> np.ndarray:
        """算一段音频的声纹向量。同步且吃 CPU，扔线程里。"""
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if rate != MODEL_RATE:
            idx = np.linspace(0, len(x) - 1, int(len(x) * MODEL_RATE / rate))
            x = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        result = await asyncio.to_thread(lambda: self._model.generate(input=x))
        return np.asarray(result[0]["spk_embedding"]).ravel()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """攒当前这句的音频，听写出来时按声纹判定。"""
        await super().process_frame(frame, direction)

        if self._mode == "off":
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._pcm.clear()
        elif isinstance(frame, InputAudioRawFrame):
            if len(self._pcm) < MODEL_RATE * 2 * 30:
                self._pcm.extend(frame.audio)
                self._rate = frame.sample_rate
        elif (
            isinstance(frame, TranscriptionFrame)
            and frame.text.strip()
            and not await self._accept(frame.text.strip())
        ):
            return

        await self.push_frame(frame, direction)

    async def _accept(self, text: str) -> bool:
        """判定这句话要不要放行，顺带记录。"""
        secs = len(self._pcm) / (self._rate * 2)
        if secs < self._min_secs:
            # 太短，向量不可靠。放行——漏进一句远比吃掉用户自己的话轻。
            self._record(text, secs, score=None, verdict="太短，放行")
            return True

        await self._ensure_model()
        vec = await self._embed(bytes(self._pcm), self._rate)

        if self._mode == "enroll":
            if secs < ENROLL_MIN_SECS:
                logger.info(
                    f"{self}: 「{text[:16]}」只有 {secs:.1f} 秒，太短不收——"
                    f"注册请说满 {ENROLL_MIN_SECS:.0f} 秒的整句"
                )
                self._record(text, secs, score=None, verdict="注册太短，未收")
                return True
            if not self._profile.add(vec):
                logger.warning(
                    f"{self}: 「{text[:16]}」跟已注册的差太远，没收下。"
                    f"如果这是你本人说的，说明前面某条注册样本有问题，"
                    f"删掉 speaker_profile.npz 重来"
                )
                self._record(text, secs, score=None, verdict="注册离群，未收")
                return True
            n = len(self._profile)
            logger.info(f"{self}: 已注册 {n}/{ENROLL_TARGET} 条「{text[:20]}」")
            if n >= ENROLL_TARGET:
                logger.info(f"{self}: 注册够了，改用 SPEAKER=observe 先看真实分数")
            self._record(text, secs, score=None, verdict=f"注册第 {n} 条")
            return True

        if not len(self._profile):
            if not self._warned:
                logger.warning(f"{self}: 还没注册声纹，全部放行。先跑 SPEAKER=enroll")
                self._warned = True
            self._record(text, secs, score=None, verdict="未注册，放行")
            return True

        score = self._profile.score(vec)
        ok = score >= self._threshold
        if self._mode == "observe":
            # 只打分不拦。阈值必须用**真实**的本人样本和**真实**的旁人样本一起标，
            # 在拿到旁人样本之前开拦截就是拿用户的话赌运气。
            self._record(text, secs, score, "观测")
            logger.info(f"{self}: 声纹 {score:.3f}（观测，不拦）「{text[:20]}」")
            return True
        self._record(text, secs, score, "放行" if ok else "拦下")
        if not ok:
            logger.info(f"{self}: 声纹 {score:.3f} 不是注册者，拦下「{text[:20]}」")
        return ok

    def _record(
        self, text: str, secs: float, score: float | None, verdict: str
    ) -> None:
        clip = self._save_audio(verdict) if SAVE_AUDIO else None
        with self._log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "at": time.time(),
                        "text": text,
                        "secs": round(secs, 2),
                        "score": round(score, 4) if score is not None else None,
                        "verdict": verdict,
                        "clip": clip,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def _save_audio(self, verdict: str) -> str | None:
        """把这一段原样存成 wav，文件名带判定结果。

        存的是**送进模型之前**的原始采样率音频，不是重采样后的——以后想换模型、换
        重采样方式重跑都还原得回来。
        """
        if not self._pcm:
            return None
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{verdict.replace('，', '_')}.wav"
        path = CORPUS_DIR / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self._rate)
            w.writeframes(bytes(self._pcm))
        return str(path.relative_to(HERE))


def build_speaker_gate() -> SpeakerGate:
    """按环境变量装配。``SPEAKER`` 取 ``off``（默认）/ ``enroll`` / ``observe`` / ``verify``。"""
    return SpeakerGate(mode=os.getenv("SPEAKER", "off"))


def stats() -> None:
    """看真实分布，用来决定阈值该不该动。"""
    if not LOG_PATH.exists():
        print("还没有记录。先跑一段对话。")
        return
    rows = [json.loads(line) for line in LOG_PATH.open(encoding="utf-8")]
    scored = [r for r in rows if r["score"] is not None]
    print(f"共 {len(rows)} 条，其中 {len(scored)} 条判过分\n")
    print(f"{'相似度':>8}{'时长':>8}{'结果':>8}  文本")
    for r in scored[-30:]:
        print(f"{r['score']:8.3f}{r['secs']:7.1f}s{r['verdict']:>8}  {r['text'][:30]}")
    if scored:
        v = sorted(r["score"] for r in scored)
        print(f"\n最低 {v[0]:.3f}   中位 {v[len(v) // 2]:.3f}   最高 {v[-1]:.3f}")
        print(f"当前阈值 {THRESHOLD}——低于它的都会被拦。")
        print("如果你自己的话有落在阈值下面的，调低 SPEAKER_THRESHOLD 或者补注册几条。")
    short = [r for r in rows if r["verdict"] == "太短，放行"]
    if short:
        print(f"\n另有 {len(short)} 条因为太短（<{MIN_SECS}s）直接放行，没判。")


if __name__ == "__main__":
    if "--stats" in sys.argv:
        stats()
    else:
        print(__doc__)
