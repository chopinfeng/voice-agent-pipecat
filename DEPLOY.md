# 部署

这套东西目前**只在 macOS / Apple Silicon 上跑过**。下面的资源数字、延迟数字全部来自
M 系列芯片，Linux x86 上没验证过。依赖本身是跨平台的（FunASR / Piper / Silero VAD /
torch），但「能装上」和「跑得动」是两回事，第一次上服务器要留出验证时间。

## 先看清楚四个拦路点

前两个是硬的，不改就上会出事。

### 一、每个连接各加载一份模型

`voice_bot.run_bot()` 里 FunASR 和 Piper 都是**按连接构造**的。FunASR 模型 2.6GB，
两个人同时连接就是 5GB 起，第三个上来大概率 OOM。

这不是理论风险——本地测试时六条场景各建一份模型，跑到第四条听写就从 0.7 秒退化到
62 秒，第五条干脆没输出。

**单用户可以先不管；多用户必须先改成进程级共享**（模型加载一次，连接复用）。这是
架构改动不是配置。

### 二、WebRTC 没配 NAT 穿透

`SmallWebRTC` 支持 `ice_servers`，但现在一个 STUN/TURN 都没配——localhost 用不上，
公网就不行了。用户在 NAT 后面直接连不上。

要么自己部 coturn，要么换托管的 WebRTC（Daily 等，pipecat 有现成 transport）。

### 三、agent 后端要外部程序

默认 `AGENT_BACKEND=claude` 会起 `claude` CLI 子进程，服务器上要装 Node 和那个 CLI。
`dsh` 后端要 `npx`。**只有 `builtin` 是纯 Python 的**，但它评测分数最低
（见 `agent_model_bench.py` 的结果）。

镜像里装不了 CLI 的话，就得接受 `builtin` 的质量，或者把 agent 拆成单独的服务。

### 四、状态是本地文件

`speaker_profile.npz`（声纹档案）和 `memory_store.jsonl`（长期记忆）都是单机文件。
单实例挂个持久卷就行；多实例或者容器重建会丢，要换外部存储。

## 资源

```
磁盘   模型约 3G（FunASR 2.6G + Piper 175M），首次启动自动下载
内存   单连接 2-3G，FunASR 是大头
CPU    不需要 GPU。SenseVoiceSmall 是非自回归的，CPU 够用
       但核数要够——实测负载一高，听写退化几十倍
Python 3.11+（本地用的 3.12）
```

**CPU 余量比绝对性能重要。**这套东西对负载极其敏感：本机负载到 44 时，本地听写从
0.7 秒变成 68.9 秒，所有延迟判据全部失真。服务器上要留足余量，并且监控负载。

## 起服务

```bash
uv sync --project pipecat
uv run --project pipecat python voice_bot.py -t webrtc --host 0.0.0.0 --port 7860
```

默认端口 7860。`/status` 是 pipecat 自带的状态端点，`/api/progress` 是我们加的后台
任务进度（见 `progress_api.py`）。

**别用按端口 kill 来重启。**没绑上端口的僵尸进程会漏网，照样占着 2.6G 的模型——
本地因为这个踩过坑，两个残留实例把负载堆到 44。用：

```bash
pkill -f voice_bot.py
```

## 密钥

`.env` 里现在有三个 key（OpenRouter、DeepSeek、火山 ARK）。**别打进镜像**，用环境变量
或密钥管理注入。`.env` 已经在 `.gitignore` 里。

必填的只有一个：

| 变量 | 说明 |
|---|---|
| `OPENROUTER_API_KEY` | 对话模型、联网查询、agent 都走它 |

可选：`DEEPSEEK_API_KEY`（给 dsh 后端用官方 provider）。

## 常用配置

全部有默认值，不配也能跑。这几个是实际会调的：

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENROUTER_MODEL` | `qwen/qwen3-max` | 对话模型。选它是因为**带工具时的尾部延迟**最稳（中位 3.48s、最慢 4.55s），更快的候选要么泄漏工具协议要么选错工具 |
| `CLAUDE_AGENT_MODEL` | `deepseek/deepseek-v4-pro` | agent 模型，评测 14/14 且最便宜 |
| `AGENT_BACKEND` | `claude` | `builtin` 纯 Python 但质量低，`dsh` 目前上游有 OOM 问题 |
| `AGENT_CONCURRENCY` | `4` | 同时跑几个后台任务。claude 后端下每个是独立子进程，抢的是本机 CPU |
| `VAD_STOP_SECS` | `0.8` | 静音多久算说完。决定听写切几段，调小会把句子切碎 |
| `SPEECH_TIMEOUT` | `0.4` | 轮次窗口。跟听写并行，已经被吸收，再调小无用 |
| `DIARIZE` | `1` | 说话人标注。给每句话加 `[主用户]` 前缀交给模型判断 |
| `SPEAKER` | `off` | 声纹拦截。**别直接开** —— 阈值必须在你自己的麦克风上标定，见 `speaker_id.py` |
| `ADDRESSEE` | `off` | 受话人判定。同样别直接开，音量阈值换设备就废 |
| `PROJECT_PATH` | `.` | agent 能读的目录，工具被限制在这里面 |

`SPEAKER` 和 `ADDRESSEE` 默认关着是有来由的：音量阈值用合成音频标的，换成真实麦克风
后**卡在了真实语音的中位数之上，11 句拦掉 8 句**，用户说话没反应。要开先按各自模块
文档里的方法在目标设备上重新标定。

## 上线前跑一遍

```bash
evals/run.sh
```

六条端到端场景，每条对应一个真实故障（打断丢记忆、异步结果送不达、任务不可见、
算数不派后台、编造、内部标记泄漏）。脚本会先体检机器负载和残留进程，不干净直接拒跑。

**注意这套目前在部署环境没验证过**——它需要合成用户语音（Kokoro，337M）和判定模型
（走 DeepSeek API）。CI/服务器上第一次跑要留时间。

## 还没做的

- **模型进程级共享**（拦路点一），多用户前必须做
- **NAT 穿透**（拦路点二），公网前必须做
- **Linux 上的完整验证**，包括资源占用和延迟基线
- **CI 配置**，等评测在部署环境跑通之后再定
