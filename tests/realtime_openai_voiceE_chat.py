import websocket
import pyaudio
import numpy as np
import json
import threading
import os

import asyncio
from openai import AsyncOpenAI


async def main():
    client = AsyncOpenAI()

    async with client.beta.realtime.connect(
        model="gpt-4o-mini-realtime-preview"
    ) as connection:
        await connection.session.update(session={"modalities": ["text"]})

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Say hello!"}],
            }
        )
        await connection.response.create()

        async for event in connection:
            if event.type == "response.text.delta":
                print(event.delta, flush=True, end="")

            elif event.type == "response.text.done":
                print()

            elif event.type == "response.done":
                break


asyncio.run(main())
