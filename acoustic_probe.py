"""验声学信号：旁人在远处说话，音量这条线索到底有多少区分力。

上一轮的测试里声学信号一次都没起作用——所有台词都是同一个 Piper 嗓子、同样音量
合成的，音量分布完全重叠。但真实场景里「用户冲着麦克风说」和「两米外两个人聊天」
音量差得很明显，这才是 DDSD 里声学模态强过文本的原因。

这里按增益缩放模拟距离（近讲 1.0、中距 0.35、远处 0.12），再叠一点底噪，看：

* RMS 能不能把远近分开，当前 0.02 这个阈值定得合不合理；
* 文本判据分不开的那些边界句，配上音量之后能不能分开。

用增益模拟距离是个简化——真实远场还有混响、直达声与反射声比例变化、频响衰减，
这些都测不出来。所以结论只能说明「音量这一维有没有用」，不能代表完整的远场效果。

运行：
    uv run --project pipecat python acoustic_probe.py
"""

import asyncio
import random
import statistics
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from addressee import AddresseeGate, _rms

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

import voice_bot as V  # noqa: E402
from filler import synthesize_clips  # noqa: E402

SEED = 20260812
# (标签, 增益, 底噪幅度)。增益越小越远。
CONDITIONS = [
    ("近讲", 1.00, 0.001),
    ("中距", 0.35, 0.002),
    ("远处", 0.12, 0.003),
]

TO_ASSISTANT = [
    "帮我看一下这个项目的整体架构",
    "这个功能是怎么实现的",
    "停一下，先别查了",
]
TO_HUMAN = [
    "中午吃什么，还是老地方",
    "你昨天看那个球赛了吗",
    "他说下周要去出差呢",
]


def scale(pcm: bytes, gain: float, noise: float, rng: random.Random) -> bytes:
    """按增益缩放并叠底噪，模拟不同距离下拾到的音频。"""
    a = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    a = a * gain
    if noise:
        # 固定种子的高斯底噪，每次跑一样。
        gen = np.random.default_rng(rng.randrange(2**32))
        a = a + gen.normal(0, noise, len(a)).astype(np.float32)
    a = np.clip(a, -1.0, 1.0)
    return (a * 32767).astype(np.int16).tobytes()


def measure(text: str, gain: float, noise: float, rng: random.Random) -> tuple[float, float]:
    """返回 (RMS, 纯文本规则分)。"""
    clips, rate = synthesize_clips(
        download_dir=V.MODEL_DIR, voice=V.TTS_VOICE, phrases=[text]
    )
    pcm = scale(clips[0], gain, noise, rng)

    gate = AddresseeGate(log_path=HERE / "logs" / "_acoustic.jsonl")
    gate._pcm = bytearray(pcm)
    gate._rate = rate
    return _rms(pcm), gate._judge(text).score


async def main():
    rng = random.Random(SEED)
    (HERE / "logs" / "_acoustic.jsonl").unlink(missing_ok=True)

    print(f"{'条件':>5}{'增益':>7}{'RMS 中位':>11}{'RMS 范围':>18}")
    print("-" * 44)
    by_cond: dict[str, list[float]] = {}
    rows = []
    for label, gain, noise in CONDITIONS:
        vals = []
        for text in TO_ASSISTANT + TO_HUMAN:
            rms, rule = measure(text, gain, noise, rng)
            vals.append(rms)
            rows.append((label, text, gain, rms, rule))
        by_cond[label] = vals
        print(
            f"{label:>5}{gain:>7.2f}{statistics.median(vals):>11.4f}"
            f"{min(vals):>10.4f}-{max(vals):.4f}"
        )

    near, far = by_cond["近讲"], by_cond["远处"]
    gap = min(near) - max(far)
    print(
        f"\n近讲与远处：{'完全分开' if gap > 0 else '有重叠'}"
        f"（间隔 {gap:+.4f}）"
    )
    thr = V and 0.02
    hit = sum(1 for v in far if v < thr)
    print(f"当前 0.02 这个阈值：远处 {hit}/{len(far)} 句判为低音量，近讲 "
          f"{sum(1 for v in near if v < thr)}/{len(near)} 句被误判")

    # 边界句：文本分不开的，加上音量能不能分开
    print("\n文本分不开的边界句，配上音量之后：")
    for label, text, _gain, rms, rule in rows:
        if 0.30 <= rule <= 0.45:
            print(f"  [{label}] rms={rms:.4f} 文本分={rule:.2f}  {text}")


if __name__ == "__main__":
    asyncio.run(main())
