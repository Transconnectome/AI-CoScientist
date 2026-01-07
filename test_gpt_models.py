
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

MODELS_TO_TEST = [
    "gpt-5-chat-latest",
    "gpt-5-pro",
    "gpt-5-pro-2025-10-06",
    "gpt-5-codex",
    "gpt-5",
    "gpt-5-2025-08-07"
]

async def test_model(client, model_name):
    print(f"\nExample testing for: {model_name}")
    print("-" * 40)
    
    success = False

    # 1. Try Chat Completions
    try:
        print(f"   [Chat API] Connecting...")
        # Some models require max_completion_tokens instead of max_tokens
        params = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if "gpt-5-2025" in model_name or "gpt-5" == model_name:
             params["max_completion_tokens"] = 10
        else:
             params["max_tokens"] = 10

        response = await client.chat.completions.create(**params)
        print(f"   ✅ [Chat API] SUCCESS! Response: {response.choices[0].message.content}")
        success = True
    except Exception as e:
        error_msg = str(e)
        if "unsupported_parameter" in error_msg and "max_tokens" in error_msg:
             print("   ⚠️ [Chat API] Retrying with max_completion_tokens...")
             try:
                 params = {
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": 10
                 }
                 response = await client.chat.completions.create(**params)
                 print(f"   ✅ [Chat API] SUCCESS (Retry)! Response: {response.choices[0].message.content}")
                 success = True
             except Exception as e2:
                 print(f"   ❌ [Chat API] Failed (Retry): {e2}")
        elif "404" in error_msg and "responses" in error_msg:
             print(f"   ⚠️ [Chat API] Failed: Model requires 'responses' API endpoint.")
        else:
             print(f"   ❌ [Chat API] Failed: {error_msg}")

    # 2. Try Responses API (if failed or just to check availability if supported)
    if hasattr(client, 'responses'):
        try:
            print(f"   [Responses API] Connecting...")
            response = await client.responses.create(
                model=model_name,
                input="Hello",
                max_output_tokens=50 # Increased to meet min requirement of 16
            )
            # Inspecting response object for content
            output_text = "Content not found in simple inspect"
            if hasattr(response, 'output_text'):
                output_text = response.output_text
            elif hasattr(response, 'output'):
                output_text = response.output
            else:
                output_text = str(response)

            print(f"   ✅ [Responses API] SUCCESS! Response: {output_text}")
            success = True
        except Exception as e:
            print(f"   ❌ [Responses API] Failed: {str(e)}")
    
    if not success:
         print(f"   ⛔ Result: {model_name} is NOT WORKING with current key/setup.")
    else:
         print(f"   ✨ Result: {model_name} IS WORKING.")

async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("No API Key found in .env")
        return
        
    client = AsyncOpenAI(api_key=api_key)
    print(f"Starting test for {len(MODELS_TO_TEST)} models...\n")
    
    for model in MODELS_TO_TEST:
        await test_model(client, model)

if __name__ == "__main__":
    asyncio.run(main())
