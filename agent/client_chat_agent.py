"""
Client chat agent using OpenRouter and a get_user_info tool.
"""
from __future__ import annotations

import json
import random
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from agent.customer_repository import CustomerRepository
from agent.openrouter_client import OpenRouterClient
from agent.churn_agent import CustomerProfile, EmailService


class ClientChatAgent:
    def __init__(
        self,
        repo: CustomerRepository,
        llm: OpenRouterClient,
        email_service: EmailService,
    ) -> None:
        self.repo = repo
        self.llm = llm
        self.email_service = email_service
        self._pending_email_changes: Dict[int, Dict[str, Any]] = {}

    def _tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_user_info",
                    "description": "Fetch a user's profile by CLIENTNUM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "client_num": {"type": "integer", "description": "Customer CLIENTNUM"},
                        },
                        "required": ["client_num"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "change_email",
                    "description": "Start an email change by sending a verification code to the new email.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "client_num": {"type": "integer", "description": "Customer CLIENTNUM"},
                            "new_email": {"type": "string", "description": "New email address"},
                        },
                        "required": ["client_num", "new_email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "validate_change_email",
                    "description": "Validate a previously sent verification code and update the email.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "client_num": {"type": "integer", "description": "Customer CLIENTNUM"},
                            "code": {"type": "string", "description": "Verification code sent to new email"},
                        },
                        "required": ["client_num", "code"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "increase_credit_limit",
                    "description": (
                        "Request a credit limit increase by percentage or amount. "
                        "Gold: up to +20%. Platinum: up to +50%. Others are not eligible."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "client_num": {"type": "integer", "description": "Customer CLIENTNUM"},
                            "percent": {"type": "number", "description": "Requested increase percentage (e.g., 10)"},
                            "amount": {"type": "number", "description": "Requested increase amount in currency"},
                        },
                        "required": ["client_num"],
                    },
                },
            },
        ]

    def _build_profile_context(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        full_name = f"{profile.get('First_Name', '')} {profile.get('Last_Name', '')}".strip()
        return {
            "client_num": profile.get("CLIENTNUM"),
            "name": full_name or None,
            "email": profile.get("Email"),
            "phone": profile.get("Phone_Number"),
            "age": profile.get("Customer_Age"),
            "gender": profile.get("Gender"),
            "income_category": profile.get("Income_Category"),
            "card_category": profile.get("Card_Category"),
            "months_on_book": profile.get("Months_on_book"),
            "credit_limit": profile.get("Credit_Limit"),
            "total_trans_amt": profile.get("Total_Trans_Amt"),
            "total_trans_ct": profile.get("Total_Trans_Ct"),
            "avg_utilization_ratio": profile.get("Avg_Utilization_Ratio"),
            "months_inactive_12_mon": profile.get("Months_Inactive_12_mon"),
        }

    def _get_user_info(self, client_num: int) -> Dict[str, Any]:
        profile = self.repo.get_customer(client_num)
        if not profile:
            raise ValueError("Customer not found")
        return self._build_profile_context(profile)

    def _send_email_change_code(self, client_num: int, new_email: str) -> Dict[str, Any]:
        profile = self.repo.get_customer(client_num)
        if not profile:
            raise ValueError("Customer not found")

        if "@" not in new_email or "." not in new_email:
            return {"ok": False, "error": "Invalid email address format"}

        code = f"{random.randint(0, 999999):06d}"
        expires_at = datetime.utcnow() + timedelta(minutes=10)
        self._pending_email_changes[client_num] = {
            "code": code,
            "new_email": new_email,
            "expires_at": expires_at,
        }

        customer = CustomerProfile.from_dict(profile)
        subject = "Your email change verification code"
        body = (
            "Hello {customer_name},\n\n"
            "We received a request to change the email on your account.\n"
            f"Your verification code is: {code}\n\n"
            "This code expires in 10 minutes.\n\n"
            "If you did not request this change, please ignore this email."
        )
        result = self.email_service.send_email(
            to_email=new_email,
            subject=subject,
            body=body,
            customer=customer,
        )
        if not result.get("success"):
            return {"ok": False, "error": result.get("error", "Email send failed")}

        return {"ok": True, "message": f"Verification code sent to {new_email}"}

    def _validate_email_change(self, client_num: int, code: str) -> Dict[str, Any]:
        pending = self._pending_email_changes.get(client_num)
        if not pending:
            return {"ok": False, "error": "No pending email change for this client"}
        if datetime.utcnow() > pending["expires_at"]:
            self._pending_email_changes.pop(client_num, None)
            return {"ok": False, "error": "Verification code expired"}
        if code.strip() != pending["code"]:
            return {"ok": False, "error": "Invalid verification code"}

        updated = self.repo.update_email(client_num, pending["new_email"])
        self._pending_email_changes.pop(client_num, None)
        if not updated:
            return {"ok": False, "error": "Customer not found"}

        return {"ok": True, "message": "Email updated successfully"}

    def _increase_credit_limit(
        self,
        client_num: int,
        percent: Optional[float],
        amount: Optional[float],
    ) -> Dict[str, Any]:
        profile = self.repo.get_customer(client_num)
        if not profile:
            raise ValueError("Customer not found")

        card_category = str(profile.get("Card_Category", "")).strip().lower()
        current_limit = float(profile.get("Credit_Limit", 0.0) or 0.0)

        if card_category == "gold":
            max_percent = 20.0
        elif card_category in {"platinum", "platinium"}:
            max_percent = 50.0
        else:
            return {
                "ok": False,
                "status": "refused",
                "reason": "category_not_eligible",
                "message": "Your current category does not allow an automatic increase. Please book a rendezvous with a conseiller.",
            }

        requested_percent: Optional[float] = None
        requested_amount: Optional[float] = None

        if percent is not None:
            try:
                requested_percent = float(percent)
            except ValueError:
                requested_percent = None
        if amount is not None:
            try:
                requested_amount = float(amount)
            except ValueError:
                requested_amount = None

        if requested_percent is None and requested_amount is None:
            return {
                "ok": False,
                "status": "refused",
                "reason": "missing_request",
                "message": "Please specify the increase as a percentage or an amount.",
            }

        if requested_percent is None and requested_amount is not None:
            if current_limit <= 0:
                return {
                    "ok": False,
                    "status": "refused",
                    "reason": "invalid_limit",
                    "message": "Current credit limit is unavailable. Please contact support.",
                }
            requested_percent = (requested_amount / current_limit) * 100.0

        if requested_amount is None and requested_percent is not None:
            requested_amount = current_limit * (requested_percent / 100.0)

        if requested_percent is None or requested_amount is None:
            return {
                "ok": False,
                "status": "refused",
                "reason": "invalid_request",
                "message": "Unable to process the requested increase. Please try again.",
            }

        if requested_percent > max_percent:
            return {
                "ok": False,
                "status": "refused",
                "reason": "above_limit",
                "message": f"The requested increase exceeds the maximum allowed for your card tier (max {max_percent:.0f}%).",
            }

        new_limit = round(current_limit + requested_amount, 2)
        updated = self.repo.update_credit_limit(client_num, new_limit)
        if not updated:
            return {"ok": False, "status": "refused", "reason": "not_found", "message": "Customer not found"}

        return {
            "ok": True,
            "status": "approved",
            "old_limit": current_limit,
            "new_limit": new_limit,
            "requested_percent": round(requested_percent, 2),
            "requested_amount": round(requested_amount, 2),
            "message": f"Approved. New credit limit is ${new_limit:,.0f}.",
        }

    def _parse_increase_request(self, message: str) -> Tuple[Optional[float], Optional[float]]:
        text = message.lower()
        percent = None
        amount = None

        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|percentage)", text)
        if percent_match:
            try:
                percent = float(percent_match.group(1))
            except ValueError:
                percent = None

        amount_match = re.search(r"\$?\s*(\d+(?:\.\d+)?)\s*(dollars|usd)?", text)
        if amount_match and ("$" in text or "dollar" in text or "usd" in text):
            try:
                amount = float(amount_match.group(1))
            except ValueError:
                amount = None

        return percent, amount

    def _format_credit_limit_response(self, result: Dict[str, Any]) -> str:
        status = result.get("status")
        if status == "approved":
            new_limit = result.get("new_limit")
            requested_percent = result.get("requested_percent")
            if new_limit is not None and requested_percent is not None:
                return (
                    f"Approved. Your credit limit is now ${float(new_limit):,.0f} "
                    f"(+{float(requested_percent):.0f}%)."
                )
            if new_limit is not None:
                return f"Approved. Your credit limit is now ${float(new_limit):,.0f}."
            return "Approved. Your credit limit has been updated."

        if status == "refused":
            message = result.get("message")
            if message:
                return message
            return "This increase request was refused."

        message = result.get("message")
        return message or "Unable to process the credit limit request."

    def _coerce_history(self, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        if not history:
            return []
        messages: List[Dict[str, Any]] = []
        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                messages.append({"role": role, "content": content})
        return messages

    def _handle_tool_call(self, name: str, args: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        try:
            if name == "get_user_info":
                client_num = int(args.get("client_num"))
                return self._get_user_info(client_num), None
            if name == "change_email":
                client_num = int(args.get("client_num"))
                new_email = str(args.get("new_email", "")).strip()
                return self._send_email_change_code(client_num, new_email), None
            if name == "validate_change_email":
                client_num = int(args.get("client_num"))
                code = str(args.get("code", "")).strip()
                return self._validate_email_change(client_num, code), None
            if name == "increase_credit_limit":
                client_num = int(args.get("client_num"))
                percent = args.get("percent")
                amount = args.get("amount")
                return self._increase_credit_limit(client_num, percent, amount), None
        except Exception as exc:
            return None, {"error": str(exc)}
        return None, {"error": f"Unknown tool: {name}"}

    async def reply(
        self,
        client_num: int,
        message: str,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        if not self.llm.is_configured():
            raise RuntimeError("OpenRouter not configured")

        system_prompt = (
            "You are Serfy Bank's client service assistant. "
            "Use tools when needed to look up profile info, change email, or increase credit limit. "
            "For credit limit changes, always call increase_credit_limit and use its response. "
            "Use only the returned profile data. Do not guess or invent data. "
            "Keep responses concise and helpful."
        )

        messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self._coerce_history(history))
        messages.append({"role": "user", "content": message})
        messages.append({"role": "user", "content": f"CLIENTNUM: {client_num}"})

        data = await self.llm.chat_raw(
            messages=messages,
            tools=self._tool_definitions(),
            tool_choice="auto",
        )

        choice = data["choices"][0]["message"]
        tool_calls = choice.get("tool_calls") or []

        if not tool_calls:
            if "credit limit" in message.lower() and "increase" in message.lower():
                percent, amount = self._parse_increase_request(message)
                result = self._increase_credit_limit(client_num, percent, amount)
                return self._format_credit_limit_response(result)
            return choice.get("content", "").strip()

        messages.append(choice)
        tool_results: List[Tuple[str, Dict[str, Any]]] = []
        for call in tool_calls:
            name = call["function"]["name"]
            args = json.loads(call["function"]["arguments"] or "{}")
            tool_response, error = self._handle_tool_call(name, args)
            if error:
                tool_response = error
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(tool_response),
                }
            )
            if isinstance(tool_response, dict):
                tool_results.append((name, tool_response))

        for name, result in tool_results:
            if name == "increase_credit_limit":
                return self._format_credit_limit_response(result)

        final = await self.llm.chat_raw(messages=messages)
        return final["choices"][0]["message"]["content"].strip()
