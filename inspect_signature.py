
import openai
from openai import AsyncOpenAI
import inspect

async def inspect_signature():
    client = AsyncOpenAI(api_key="dummy")
    print("\nSignature of client.responses.create:")
    try:
        sig = inspect.signature(client.responses.create)
        print(sig)
    except Exception as e:
        print(f"Could not get signature: {e}")
        # Try help
        help(client.responses.create)

import asyncio
asyncio.run(inspect_signature())
