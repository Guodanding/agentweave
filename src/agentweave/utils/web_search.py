import logging
import os
from typing import Optional

import aiohttp
import json

logger = logging.getLogger(__name__)


class BraveLLMContextClient:

    BASE_URL = "https://api.search.brave.com/res/v1/llm/context"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def get_context(
        self,
        query: str,
        country: Optional[str] = None,
        search_lang: Optional[str] = None,
        count: Optional[int] = None,
        max_urls: Optional[int] = 8,
        max_tokens: Optional[int] = 2048,
        max_snippets: Optional[int] = None,
        threshold_mode: Optional[str] = None,
        max_tokens_per_url: Optional[int] = None,
        max_snippets_per_url: Optional[int] = None,
    ) -> dict:
        headers = {
            "x-subscription-token": self.api_key,
            "Accept": "application/json",
        }

        params = {"q": query}

        if country is not None:
            params["country"] = country
        if search_lang is not None:
            params["search_lang"] = search_lang
        if count is not None:
            params["count"] = count
        if max_urls is not None:
            params["maximum_number_of_urls"] = max_urls
        if max_tokens is not None:
            params["maximum_number_of_tokens"] = max_tokens
        if max_snippets is not None:
            params["maximum_number_of_snippets"] = max_snippets
        if threshold_mode is not None:
            params["context_threshold_mode"] = threshold_mode
        if max_tokens_per_url is not None:
            params["maximum_number_of_tokens_per_url"] = max_tokens_per_url
        if max_snippets_per_url is not None:
            params["maximum_number_of_snippets_per_url"] = max_snippets_per_url

        async with aiohttp.ClientSession() as session:
            async with session.get(
                self.BASE_URL,
                headers=headers,
                params=params,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"❌API Error {response.status}: {error_text}")

    async def web_search(
        self,
        query: str,
        top_k: int,
        request_id: str = None,
        question_idx: int = None,
    ) -> str:
        log_prefix = f"[Req {request_id}] Question {question_idx + 1}" if request_id is not None else "Web search"

        words = query.split()
        if len(words) > 50:
            query = " ".join(words[:50])
            logger.info(f"{log_prefix} - Query truncated to 50 words")

        try:
            result = await self.get_context(query=query)
            result_str = json.dumps(result, ensure_ascii=False)
            logger.info(f"{log_prefix} - Web search API success, len={len(result_str)}")
            return result_str

        except Exception as e:
            error_msg = str(e) if str(e) else f"{type(e).__name__}"
            logger.error(f"{log_prefix} - Web search error: {error_msg}")
            return None


_search_client: Optional[BraveLLMContextClient] = None


def get_search_client() -> Optional[BraveLLMContextClient]:
    global _search_client
    if _search_client is None:
        api_key = os.getenv("BRAVE_API_KEY")

        if not api_key:
            raise ValueError("BRAVE_API_KEY environment variable not set")

        _search_client = BraveLLMContextClient(api_key)
    return _search_client
