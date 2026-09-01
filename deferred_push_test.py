"""异步工具结果：用户说话时落地的那一轮推理，等他说完补不补得回来。

这个测试锁的是 ``filters.RememberInterrupted`` 里那半个补丁，也是它存在的唯一理由。

上游 ``_handle_function_call_result`` 结尾是：

    if run_llm and not self._user_speaking:
        await self._maybe_push_context_after_function_result()

结果本身不会丢——``_handle_function_call_finished`` 已经无条件把它作为一条 developer
消息写进上下文了（``async_tool_messages`` 协议）。丢的是**主动出声的那一次推理**。
而 bot 在说话时上游留了补跑标记（``_push_context_on_bot_stopped_speaking``），
用户在说话时什么都不留，``UserStoppedSpeakingFrame`` 只把标志位置回 False 就完了。

所以这里跑两遍同一段序列：上游原版和打了补丁的，比它们补跑了几次。**对照组是必须的**
——只测补丁通过说明不了问题，得看见原版确实不补跑，才证明差别来自补丁。

跑：

    uv run --project pipecat python deferred_push_test.py

**如果哪天原版那一行也变成 1 次，说明上游把这个不对称修了**，那 ``RememberInterrupted``
里的 ``_push_context_on_user_stopped_speaking`` 那几个覆盖就该删掉。这个测试会先失败
提醒你，别让它一直挂着。
"""

import asyncio
import sys

from loguru import logger
from pipecat.frames.frames import (
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection

from filters import RememberInterrupted

TOOL_ID = "call-1"


async def drive(cls) -> tuple[int, int, bool]:
    """跑一遍「工具在跑 → 用户开口 → 结果落地 → 用户说完」。

    Args:
        cls: 要测的聚合器类。

    Returns:
        (结果落地时补跑次数, 用户说完后累计补跑次数, 结果有没有进上下文)。
    """
    context = LLMContext()
    pair = LLMContextAggregatorPair(context, user_params=LLMUserAggregatorParams())
    agg = cls(
        context,
        params=LLMAssistantAggregatorParams(),
        _paired_user_aggregator=pair.user(),
    )

    # 不建整条管线：这里只关心 push_context_frame 被调了几次，下游推什么无所谓。
    pushes: list[FrameDirection] = []
    original = agg.push_context_frame

    async def counting(direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushes.append(direction)
        await original(direction)

    agg.push_context_frame = counting
    agg.push_frame = lambda frame, direction=None: asyncio.sleep(0)

    # cancel_on_interruption=False 才是异步工具，结果走 developer 消息那条路。
    await agg.process_frame(
        FunctionCallInProgressFrame(
            function_name="ask_project",
            tool_call_id=TOOL_ID,
            arguments={"question": "填充语怎么做的"},
            cancel_on_interruption=False,
        ),
        FrameDirection.DOWNSTREAM,
    )
    await agg.process_frame(UserStartedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await agg.process_frame(
        FunctionCallResultFrame(
            function_name="ask_project",
            tool_call_id=TOOL_ID,
            arguments={},
            result={"answer": "填充语放在 TTS 之后"},
        ),
        FrameDirection.DOWNSTREAM,
    )
    during = len(pushes)

    await agg.process_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
    await asyncio.sleep(0)

    settled = any(
        "async_tool" in str(m.get("content", "")) for m in context.get_messages()
    )
    return during, len(pushes), settled


async def main() -> int:
    base_during, base_after, base_settled = await drive(LLMAssistantAggregator)
    fix_during, fix_after, fix_settled = await drive(RememberInterrupted)

    print(
        f"上游原版             结果落地时 {base_during} 次，用户说完后累计 {base_after} 次，"
        f"结果进上下文={base_settled}"
    )
    print(
        f"RememberInterrupted  结果落地时 {fix_during} 次，用户说完后累计 {fix_after} 次，"
        f"结果进上下文={fix_settled}"
    )

    problems = []
    if not (base_settled and fix_settled):
        problems.append("结果没进上下文——那前提就不成立，先去看 async_tool_messages")
    if base_during or fix_during:
        problems.append("用户还在说话时就补跑了，会抢他的话")
    if base_after:
        problems.append("上游原版也补跑了：这个不对称可能已被上游修复，补丁该删了")
    if fix_after != 1:
        problems.append(f"补丁没补跑：期望 1 次，实际 {fix_after}")

    for p in problems:
        print(f"✗ {p}")
    print("PASS" if not problems else "FAIL")
    return 1 if problems else 0


if __name__ == "__main__":
    logger.remove()
    sys.exit(asyncio.run(main()))
