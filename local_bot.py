"""Fully local Pipecat bot — no cloud API keys required.

Speaks a greeting through the local Piper TTS model when a browser client
connects over WebRTC. Use it to confirm the runner, transport and audio output
path all work on this machine.

Run with:
    uv run --project pipecat python local_bot.py -t webrtc

Then open http://localhost:7860/client/ and click Connect.
"""

from pathlib import Path

from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

MODEL_DIR = Path(__file__).parent / "piper-voices"

transport_params = {
    "webrtc": lambda: TransportParams(audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    tts = PiperTTSService(
        download_dir=MODEL_DIR,
        settings=PiperTTSService.Settings(voice="en_US-ryan-high"),
    )

    worker = PipelineWorker(
        Pipeline([tts, transport.output()]),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected, speaking greeting")
        await worker.queue_frames(
            [
                TTSSpeakFrame(
                    "Hello. Pipecat is running locally on your machine, "
                    "with speech synthesized by Piper. No API keys needed."
                ),
                EndFrame(),
            ]
        )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
