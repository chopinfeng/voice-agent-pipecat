"""训一个融合文本和声学的受话人分类器，交叉验证看它比手调权重强多少。

前两步各自量出了边界：

* 文本判据（关键词 93%、LLM 79%、两者融合 100%）判的是**意图指向**——这句话冲谁说的。
* 音量判的是**距离**——近讲 RMS 0.13-0.19、远处 0.016-0.024，完全分开。但同样近讲时，
  「这个功能怎么实现」和「中午吃什么」音量一模一样，它对意图一无所知。

两者是正交的两维，手写权重去配比很难调准。这里直接让逻辑回归自己学系数——样本少，
用留一交叉验证，别拿训练集上的分数骗自己。

标签取的是**该不该响应**，不是单纯的「对谁说」：远场里对着助手喊也算不该响应，
因为那种情况下听写本来就不可靠。

运行：
    uv run --project pipecat python addressee_train.py
"""

import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut

from acoustic_probe import CONDITIONS, scale
from addressee import _DOMAIN, _TO_ASSISTANT, _TO_HUMAN, _rms

HERE = Path(__file__).parent
load_dotenv(HERE / ".env", override=True)

import voice_bot as V  # noqa: E402
from filler import synthesize_clips  # noqa: E402

SEED = 20260812

# 对助手说的
ASSISTANT_LINES = [
    "帮我看一下这个项目的整体架构",
    "延迟测试那部分是怎么做的",
    "你能把刚才的结果再说一遍吗",
    "查一下依赖了哪些第三方库",
    "停一下，先别查了",
    "这个功能是怎么实现的",
    "再说一遍",
    "把结果讲简单点",
    "现在查到哪一步了",
    "这个文件有多少行",
]
# 旁人之间聊的
HUMAN_LINES = [
    "你昨天看那个球赛了吗",
    "我觉得楼下那家咖啡还行吧",
    "他说下周要去出差呢",
    "哎呀这个天气真是够热的",
    "中午吃什么，还是老地方",
    "你觉得呢，我是无所谓啦",
    "那我先走了啊，回头聊",
    "这个周末有什么安排没有",
    "老板刚才在会上说什么了",
    "我昨天睡得特别晚",
]

FEATURES = ["to_assistant", "to_human", "domain", "has_you", "question", "rms", "secs"]


def featurize(text: str, pcm: bytes, rate: int) -> list[float]:
    """把一句话和它的音频变成特征向量。"""
    return [
        float(min(len(_TO_ASSISTANT.findall(text)), 2)),
        float(min(len(_TO_HUMAN.findall(text)), 2)),
        float(min(len(_DOMAIN.findall(text)), 2)),
        float("你" in text),
        float(text.endswith(("？", "?", "吗", "呢"))),
        _rms(pcm),
        len(pcm) / (rate * 2),
    ]


def build_dataset() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """台词 × 距离条件，合成音频后抽特征。

    Returns:
        (特征矩阵, 标签, 每条样本的说明)。
    """
    rng = random.Random(SEED)
    lines = [(t, 1) for t in ASSISTANT_LINES] + [(t, 0) for t in HUMAN_LINES]
    texts = [t for t, _ in lines]
    clips, rate = synthesize_clips(
        download_dir=V.MODEL_DIR, voice=V.TTS_VOICE, phrases=texts
    )

    X, y, notes = [], [], []
    for (text, to_assistant), clip in zip(lines, clips, strict=True):
        for label, gain, noise in CONDITIONS:
            pcm = scale(clip, gain, noise, rng)
            X.append(featurize(text, pcm, rate))
            # 该响应 = 对助手说 且 不是远场。远场里连听写都不可靠，
            # 与其答错不如不答。
            y.append(int(to_assistant and label != "远处"))
            notes.append(f"[{label}] {text}")
    return np.array(X), np.array(y), notes


def main():
    X, y, notes = build_dataset()
    print(f"样本 {len(y)} 条（{y.sum()} 条该响应 / {len(y) - y.sum()} 条不该）\n")

    # 留一交叉验证：样本少，每条都当一次测试集。
    loo = LeaveOneOut()
    correct, wrong = 0, []
    for train_idx, test_idx in loo.split(X):
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])[0]
        if pred == y[test_idx][0]:
            correct += 1
        else:
            wrong.append((notes[test_idx[0]], y[test_idx][0], pred))

    print(f"留一交叉验证准确率：{correct / len(y):.0%}（{correct}/{len(y)}）")
    if wrong:
        print("\n判错的：")
        for note, truth, pred in wrong:
            print(f"  真值 {truth} 判成 {pred}   {note}")

    # 全量训一遍看系数，理解它到底在依赖什么。
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    print("\n学到的权重（正=倾向该响应）：")
    for name, w in sorted(
        zip(FEATURES, clf.coef_[0], strict=True), key=lambda kv: -abs(kv[1])
    ):
        print(f"  {name:<14}{w:+.2f}")

    # 跟手调权重比一比
    from addressee import AddresseeGate

    gate = AddresseeGate(log_path=HERE / "logs" / "_train_cmp.jsonl")
    rule_correct = 0
    for (text_note, feats, truth) in zip(notes, X, y, strict=True):
        text = text_note.split("] ", 1)[1]
        gate._pcm = bytearray(int(feats[6] * 16000 * 2))  # 只为让时长对上
        gate._rate = 16000
        u = gate._judge(text)
        # 手调那版没用上音量，这里按它自己的阈值判
        rule_correct += int((u.score >= 0.35) == bool(truth))
    print(f"\n手调权重在同一批样本上：{rule_correct / len(y):.0%}")


if __name__ == "__main__":
    main()
