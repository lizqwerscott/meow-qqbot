from core.ai.protocol import StreamSnapshot


async def emit_snapshot(callbacks, text, generation=0):
    if callbacks and callbacks.on_snapshot:
        await callbacks.on_snapshot(
            StreamSnapshot(generation=generation, text=text, reasoning=None)
        )
