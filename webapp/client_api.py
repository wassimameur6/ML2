"""
API helpers for the client-facing Streamlit app.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests


API_URL = os.getenv("API_URL", "http://localhost:8080")


def _safe_json(response: requests.Response) -> Optional[Dict[str, Any]]:
    try:
        return response.json()
    except Exception:
        return None


def fetch_customer_profile(client_num: int, timeout: int = 15) -> Optional[Dict[str, Any]]:
    url = f"{API_URL}/customers/{client_num}"
    try:
        resp = requests.get(url, timeout=timeout)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return _safe_json(resp)


def send_chat_message(
    client_num: int,
    message: str,
    history: Optional[List[Dict[str, Any]]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """
    Placeholder for future agent endpoint.
    Expected future API: POST /client/chat
    Payload: { client_num: int, message: str }
    """
    url = f"{API_URL}/client/chat"
    payload = {"client_num": client_num, "message": message, "history": history or []}
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException:
        return {"ok": False, "error": "chat_unavailable"}

    if resp.status_code == 404 or resp.status_code == 501:
        return {"ok": False, "error": "chat_not_ready"}

    data = _safe_json(resp) or {}
    return {"ok": resp.ok, "data": data, "status_code": resp.status_code}
