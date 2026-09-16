#!/usr/bin/env bash
# 跑官方评测，前面加一道体检。
#
# 场景、驱动、判定全部交给 pipecat 自带的 `pipecat eval`——它比自造的那套强：
# send_after 能表达打断、function_call 能校验参数、judge 用自然语言判据而不是
# 脆弱的关键词匹配（自造版的关键词判据翻车过三次）。
#
# 这个脚本只补官方框架不管的两件事，都是被这台机器坑出来的：
#
#   1. 残留进程。每次「重启」只 kill 监听端口的那个，没绑上端口的僵尸会漏网，
#      各自占着一份 FunASR 模型。实测堆到负载 44 时本地听写从 0.7 秒变成 68.9 秒，
#      时间线顺序都乱了——那种数据比没有更糟，因为它看起来像真的。
#   2. 机器负载。跑的时候有别的程序把负载顶到 166，听写要 62 秒。延迟判据在这种
#      环境下全部失真。
#
# 用法：
#   evals/run.sh                      跑全部场景
#   evals/run.sh interrupt_storm      只跑一个
set -euo pipefail
cd "$(dirname "$0")/.."

MAX_LOAD="${E2E_MAX_LOAD:-6}"
PORT="${EVAL_PORT:-7860}"

# ---- 体检 ----
# pgrep 找不到东西时返回 1，在 set -e 下会直接掐死脚本——也就是「没有残留进程」
# 这个正常情况反而跑不起来。加 || true 兜住。
stray=$( (pgrep -f "voice_bot.py" || true) | wc -l | tr -d ' ')
if [ "$stray" != "0" ]; then
  echo "有 $stray 个残留的 voice_bot 进程，先清掉：pkill -f voice_bot.py" >&2
  exit 1
fi
load=$(uptime | sed 's/.*averages: //' | awk '{print $1}')
if [ "$(echo "$load $MAX_LOAD" | awk '{print ($1>$2)}')" = "1" ]; then
  echo "机器负载 $load 超过 $MAX_LOAD，延迟判据会失真，先等它降下来" >&2
  exit 1
fi
echo "体检通过（负载 $load，无残留进程）"

# ---- 判定模型 ----
# 官方默认判定用本机 ollama，没装；OpenAI 直连在当前网络下是 403。
# 指到 DeepSeek 官方：快、有额度，也符合「只用国产开源模型」。
set -a; . ./.env; set +a
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_BASE_URL="https://api.deepseek.com"

# ---- 起 bot ----
uv run --project pipecat python voice_bot.py -t eval --port "$PORT" > logs/eval_bot.log 2>&1 &
BOT=$!
# 起来之前不要开跑：本地模型（FunASR + Piper）加载要几十秒，抢跑会把加载时间
# 算进第一条场景的延迟里。
trap 'kill $BOT 2>/dev/null || true' EXIT
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/" >/dev/null 2>&1 && break
  sleep 2
done

# ---- 跑场景 ----
if [ $# -gt 0 ]; then
  files=()
  for n in "$@"; do files+=("evals/$n.yaml"); done
else
  files=(evals/interrupt_storm.yaml evals/search_interrupted.yaml
         evals/task_visible.yaml evals/compute_dispatch.yaml
         evals/no_fabrication.yaml evals/bystander.yaml)
fi
uv run --project pipecat pipecat eval run "${files[@]}" \
  --bot-url "ws://127.0.0.1:$PORT" -v
