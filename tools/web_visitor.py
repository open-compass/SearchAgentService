#!/usr/bin/env python3
"""
Web Visitor Tool (standalone version)
Fetches web pages via Jina Reader + extracts summaries with LLM, usable as a direct function call.
Reference: https://github.com/Alibaba-NLP/DeepResearch/blob/main/inference/tool_visit.py
"""
import asyncio
import json
import random
import time
from typing import Dict, List, Union

import aiohttp
from openai import APITimeoutError, AsyncOpenAI

# =============================================================================
# Configuration (lazy init, injected via configure())
# =============================================================================
MODEL_NAME = ""
BASE_URL = ""
API_KEY = "sk-admin"
JINA_API_KEY = ""
_configured = False

# Web fetching config
JINA_URL = "https://r.jina.ai/"
JINA_TIMEOUT = 50
JINA_MAX_RETRIES = 10

# LLM config
LLM_TIMEOUT = 1000
LLM_MAX_RETRY = 50
LLM_SLEEP_INTERVAL = 5

# Content truncation config
MAX_CONTENT_CHARS = 150000
MAX_SUMMARY_RETRIES = 3
MAX_PARSE_RETRIES = 3

# =============================================================================
# Prompt (reference: DeepResearch)
# =============================================================================
EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content**
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""

FAILED_MSG_TEMPLATE = (
    'The useful information in {url} for user goal {goal} as follows: \n\n'
    'Evidence in page: \n'
    'The provided webpage content could not be accessed. '
    'Please check the URL or file format.\n\n'
    'Summary: \n'
    'The webpage content could not be processed, and therefore, '
    'no information is available.\n\n'
)

# =============================================================================
# Text Truncation
# =============================================================================
def truncate_text(text: str, max_chars: int = MAX_CONTENT_CHARS) -> str:
    """Truncate text by character length, keeping the head."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# =============================================================================
# Web Fetching (direct Jina API call, no browse service dependency)
# =============================================================================
async def fetch_page(url: str, session: aiohttp.ClientSession) -> str:
    """Fetch web page content via Jina Reader with retries."""
    for attempt in range(JINA_MAX_RETRIES):
        try:
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Accept": "text/plain",
                "X-Return-Format": "text",
                "X-Timeout": "10",
            }
            async with session.get(
                f"{JINA_URL}{url}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=JINA_TIMEOUT),
            ) as resp:
                if resp.status == 200:
                    return await resp.text()
                print(f"[fetch_page] HTTP {resp.status} for {url}")
        except Exception as e:
            print(f"[fetch_page] attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(0.5)
    return ""

# =============================================================================
# LLM Client
# =============================================================================
class LLMClient:
    """Lightweight LLM client, depends only on openai SDK."""

    def __init__(self):
        urls = [u.strip() for u in BASE_URL.split(",") if u.strip()]
        self.clients = [
            AsyncOpenAI(api_key=API_KEY, base_url=u) for u in urls
        ]

    async def chat(self, messages: list[dict]) -> str:
        for attempt in range(LLM_MAX_RETRY):
            try:
                client = random.choice(self.clients)
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    stream=False,
                    temperature=0.7,
                    timeout=LLM_TIMEOUT,
                )
                return resp.choices[0].message.content or ""
            except (APITimeoutError, TimeoutError) as e:
                print(f"[LLM] timeout attempt {attempt + 1}: {e}")
                if attempt == LLM_MAX_RETRY - 1:
                    return ""
                await asyncio.sleep(LLM_SLEEP_INTERVAL)
            except Exception as e:
                print(f"[LLM] error attempt {attempt + 1}: {e}")
                if attempt == LLM_MAX_RETRY - 1:
                    return ""
                await asyncio.sleep(LLM_SLEEP_INTERVAL)
        return ""

# =============================================================================
# Page Content Extraction (reference: DeepResearch readpage_jina flow)
# =============================================================================
async def extract_page_info(
    url: str, goal: str, content: str, llm: LLMClient
) -> str:
    """Extract goal-related information from web page content using LLM."""
    content = truncate_text(content)
    prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
    messages = [{"role": "user", "content": prompt}]

    raw = await llm.chat(messages)

    # If response is too short, retry with progressively shorter content
    retries = MAX_SUMMARY_RETRIES
    while len(raw) < 10 and retries > 0:
        truncate_len = int(0.7 * len(content)) if retries > 1 else 25000
        print(f"[extract] retry for {url}, truncating to {truncate_len} chars")
        content = content[:truncate_len]
        prompt = EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)
        messages = [{"role": "user", "content": prompt}]
        raw = await llm.chat(messages)
        retries -= 1

    # Parse JSON
    if isinstance(raw, str):
        raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = None
    for _ in range(MAX_PARSE_RETRIES):
        try:
            parsed = json.loads(raw)
            break
        except (json.JSONDecodeError, TypeError):
            raw = await llm.chat(messages)
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()

    if parsed is None:
        return FAILED_MSG_TEMPLATE.format(url=url, goal=goal)

    return (
        f"The useful information in {url} for user goal {goal} as follows: \n\n"
        f"Evidence in page: \n{parsed.get('evidence', '')}\n\n"
        f"Summary: \n{parsed.get('summary', '')}\n\n"
    )

# =============================================================================
# Single URL Processing Flow
# =============================================================================
async def visit_single_url(
    url: str, goal: str, llm: LLMClient, session: aiohttp.ClientSession
) -> str:
    """Fetch a single URL and extract information using LLM."""
    content = await fetch_page(url, session)
    if not content:
        return FAILED_MSG_TEMPLATE.format(url=url, goal=goal)
    return await extract_page_info(url, goal, content, llm)

# =============================================================================
# Public Interface
# =============================================================================
llm: LLMClient = None
http_session: aiohttp.ClientSession = None


def configure(jina_api_key: str = "", model_name: str = "",
              base_url: str = "", api_key: str = "sk-admin", timeout: int = 1000):
    """Initialize web_visitor config, called by registry during build.

    Args:
        jina_api_key: Jina Reader API key
        model_name: LLM model name
        base_url: LLM base URL (supports comma-separated multiple URLs)
        api_key: LLM API key
        timeout: LLM request timeout in seconds
    """
    global MODEL_NAME, BASE_URL, API_KEY, JINA_API_KEY, LLM_TIMEOUT, llm, _configured

    if _configured:
        return

    MODEL_NAME = model_name
    BASE_URL = base_url
    API_KEY = api_key
    JINA_API_KEY = jina_api_key
    LLM_TIMEOUT = timeout
    llm = LLMClient()
    _configured = True


async def get_session() -> aiohttp.ClientSession:
    """Lazy-initialize aiohttp session."""
    global http_session
    if http_session is None or http_session.closed:
        http_session = aiohttp.ClientSession()
    return http_session


async def visit(url: Union[str, List[str]], goal: str) -> str:
    """Visit webpage(s) and return the summary of the content."""
    session = await get_session()

    if isinstance(url, str):
        return await visit_single_url(url, goal, llm, session)

    # Multiple URLs: process sequentially, skip remaining after 900s timeout
    results = []
    start = time.time()
    for u in url:
        if time.time() - start > 900:
            results.append(FAILED_MSG_TEMPLATE.format(url=u, goal=goal))
            continue
        try:
            r = await visit_single_url(u, goal, llm, session)
        except Exception as e:
            r = f"Error fetching {u}: {e}"
        results.append(r)

    return "\n=======\n".join(results)


async def close_session():
    """Close http session for graceful shutdown."""
    global http_session
    if http_session and not http_session.closed:
        await http_session.close()


# Tool schema (for registry, compatible with original MCP tool schema)
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "visit",
        "description": "Visit webpage(s) and return the summary of the content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
                },
                "goal": {
                    "type": "string",
                    "description": "The goal of the visit for webpage(s)."
                }
            },
            "required": ["url", "goal"]
        }
    }
}
