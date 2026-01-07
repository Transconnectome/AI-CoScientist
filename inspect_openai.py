
import openai
from openai import AsyncOpenAI
import inspect

async def inspect_client():
    client = AsyncOpenAI(api_key="dummy")
    print("Direct attributes of client:")
    for attr in dir(client):
        if not attr.startswith("_"):
            print(f"- {attr}")
    
    if hasattr(client, 'responses'):
        print("\nFOUND 'responses' attribute!")
    elif hasattr(client, 'beta') and hasattr(client.beta, 'responses'):
        print("\nFOUND 'beta.responses' attribute!")
    else:
        print("\n'responses' attribute NOT found.")

import asyncio
asyncio.run(inspect_client())
