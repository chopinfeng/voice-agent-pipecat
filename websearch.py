"""给 agent 用的联网查询命令：``python3 websearch.py "要查什么"``。

DeepSeek Harness 的 headless profile **一个联网工具都没有**（工具只有 bash / fs /
fs-search / jobs / skill），所以问它天气股价这类事，它只能凭记忆答——实测它会说
「当前网络搜索 API 没有配置密钥」，那是模型编的解释，真实原因是它压根没这个工具。

补法不是去装插件（npm 上没有官方的联网插件，技能文档 404），而是用它**确定有的**
能力：bash。这个项目本来就有联网查询（OpenRouter 的 ``:online`` 变体），包成一个
命令行摆在工作目录里，任何有 shell 的 agent 都能用，不挑后端。

Claude 那边有原生 WebSearch，用不上这个，但摆着不碍事——提示里写成「有 WebSearch
就用它，没有就跑这个命令」。

用法：
    python3 websearch.py "北京明天天气"
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def main() -> int:
    """查一次，把结果打到标准输出。"""
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print("用法：python3 websearch.py \"要查什么\"", file=sys.stderr)
        return 2

    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        # 说清楚是缺什么，别让调用方（模型）自己猜然后编一个解释。
        print("没有 OPENROUTER_API_KEY，无法联网查询。", file=sys.stderr)
        return 1

    from openai import AsyncOpenAI

    import tools

    client = AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    try:
        answer = await tools.web_search(
            query, client, os.getenv("SEARCH_MODEL", "deepseek/deepseek-v4-flash")
        )
    finally:
        await client.close()
    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
