
import os
import asyncio
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def verify_openai(api_key: str):
    logger.info("Verifying OpenAI API Key...")
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        if model in ["gpt-5-pro", "gpt-5-codex", "gpt-5-pro-2025-10-06"]:
             response = await client.responses.create(
                model=model,
                input="Hello",
                max_output_tokens=20
            )
             # Basic extraction for verification script
             content = getattr(response, 'output_text', getattr(response, 'output', str(response)))
             logger.info(f"✅ OpenAI Verification Successful ({model}): {content}")
        else:
            response = await client.chat.completions.create(
                model=model, # Use configured model
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info(f"✅ OpenAI Verification Successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logger.error(f"❌ OpenAI Verification Failed: {str(e)}")
        return False

async def verify_anthropic(api_key: str):
    logger.info("Verifying Anthropic API Key...")
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)
        response = await client.messages.create(
            model="claude-3-haiku-20240307", # Use a cheap model
            max_tokens=5,
            messages=[{"role": "user", "content": "Hello"}]
        )
        logger.info(f"✅ Anthropic Verification Successful: {response.content[0].text}")
        return True
    except Exception as e:
        logger.error(f"❌ Anthropic Verification Failed: {str(e)}")
        return False

async def verify_gemini(api_key: str):
    logger.info("Verifying Gemini API Key...")
    try:
        # Try google-generativeai package
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        response = await model.generate_content_async("Hello")
        logger.info(f"✅ Gemini Verification Successful: {response.text}")
        return True
    except ImportError:
        logger.warning("google-generativeai package not found, trying REST API fallback could be done but skipping for now.")
        return False
    except Exception as e:
        logger.error(f"❌ Gemini Verification Failed: {str(e)}")
        return False

async def verify_tavily(api_key: str):
    logger.info("Verifying Tavily API Key...")
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        # Tavily python client is synchronous usually? Let's check or run in thread if needed, but for prompt verification sync is fine
        response = client.search(query="Tesla", max_results=1)
        logger.info(f"✅ Tavily Verification Successful: Found {len(response.get('results', []))} results")
        return True
    except ImportError:
         # Fallback to requests
        import requests
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={"api_key": api_key, "query": "Tesla", "max_results": 1}
            )
            response.raise_for_status()
            logger.info(f"✅ Tavily Verification Successful (via REST): Status {response.status_code}")
            return True
        except Exception as e_rest:
            logger.error(f"❌ Tavily Verification Failed: {str(e_rest)}")
            return False
    except Exception as e:
        logger.error(f"❌ Tavily Verification Failed: {str(e)}")
        return False

async def verify_deepseek(api_key: str):
    logger.info("Verifying DeepSeek API Key...")
    try:
        from openai import AsyncOpenAI
        # DeepSeek is compatible with OpenAI SDK
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        logger.info(f"✅ DeepSeek Verification Successful: {response.choices[0].message.content}")
        return True
    except Exception as e:
        logger.error(f"❌ DeepSeek Verification Failed: {str(e)}")
        return False

async def main():
    results = {}
    
    # OpenAI
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        results["OpenAI"] = await verify_openai(openai_key)
    else:
        logger.warning("OPENAI_API_KEY not found in env")
        results["OpenAI"] = None

    # Anthropic
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        results["Anthropic"] = await verify_anthropic(anthropic_key)
    else:
        logger.warning("ANTHROPIC_API_KEY not found in env")
        results["Anthropic"] = None

    # Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        results["Gemini"] = await verify_gemini(gemini_key)
    else:
        logger.warning("GEMINI_API_KEY / GOOGLE_API_KEY not found in env")
        results["Gemini"] = None
    
    # Tavily
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        results["Tavily"] = await verify_tavily(tavily_key)
    else:
        logger.warning("TAVILY_API_KEY not found in env")
        results["Tavily"] = None

    # DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key:
        results["DeepSeek"] = await verify_deepseek(deepseek_key)
    else:
        logger.warning("DEEPSEEK_API_KEY not found in env")
        results["DeepSeek"] = None

    logger.info("\n--- Verification Summary ---")
    for service, status in results.items():
        if status is True:
            print(f"✅ {service}: OK")
        elif status is False:
            print(f"❌ {service}: FAILED")
        else:
            print(f"⚠️ {service}: NOT CONFIGURED")

if __name__ == "__main__":
    asyncio.run(main())
