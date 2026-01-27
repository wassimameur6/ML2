"""
OpenRouter client using OpenAI-compatible chat completions.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        app_url: str | None = None,
        app_title: str | None = None,
    ) -> None:
        # Use OpenAI if key exists, otherwise fall back to OpenRouter
        openai_key = os.getenv("OPENAI_API_KEY", "")
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

        if openai_key:
            self.api_key = api_key or openai_key
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            self._use_openai = True
        else:
            self.api_key = api_key or openrouter_key
            self.base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
            self._use_openai = False
        self.app_url = app_url or os.getenv("OPENROUTER_APP_URL", "http://localhost")
        self.app_title = app_title or os.getenv("OPENROUTER_APP_TITLE", "Serfy Bank Client Service")

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def chat_raw(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        timeout: int = 30,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("OpenRouter API key is missing")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Add OpenRouter-specific headers only if using OpenRouter
        if not self._use_openai:
            headers["HTTP-Referer"] = self.app_url
            headers["X-Title"] = self.app_title
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int = 300,
        timeout: int = 30,
    ) -> str:
        data = await self.chat_raw(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        return data["choices"][0]["message"]["content"].strip()
