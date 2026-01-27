"""
Clean client-facing Streamlit app with landing + chat UI.
"""
from __future__ import annotations

import base64
import os
from typing import Dict, Optional

import streamlit as st

from client_api import fetch_customer_profile, send_chat_message
from client_styles import get_client_css


def _init_state() -> None:
    if "client_num" not in st.session_state:
        st.session_state.client_num = None
    if "client_profile" not in st.session_state:
        st.session_state.client_profile = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _reset_session() -> None:
    st.session_state.client_num = None
    st.session_state.client_profile = None
    st.session_state.chat_history = []


def _get_logo_base64() -> Optional[str]:
    logo_path = os.path.join(os.path.dirname(__file__), "srf.jpeg")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def _render_hero(client_num: Optional[int]) -> None:
    chip = f"<span class='client-chip'>CLIENTNUM {client_num}</span>" if client_num else ""
    st.markdown(
        f"""
        <div class="hero">
            <h1 class="hero-title">Serfy Bank Client Service</h1>
            <div class="hero-sub">Your secure client service space for account questions, guidance, and support.</div>
            {chip}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_landing() -> None:
    col_left, col_right = st.columns([1.4, 1])
    with col_left:
        st.markdown(
            """
            <h2>Fast, private support with your CLIENTNUM</h2>
            <p class="hero-sub">
                Enter your CLIENTNUM to start a secure chat. You can check account info, update email,
                or request a credit limit increase.
            </p>
            <div class="landing-grid">
                <div class="info-card">
                    <h4>Identify</h4>
                    <p>We verify your CLIENTNUM and load your profile context.</p>
                </div>
                <div class="info-card">
                    <h4>Chat</h4>
                    <p>Ask about your card, usage, or account details.</p>
                </div>
                <div class="info-card">
                    <h4>Resolve</h4>
                    <p>Change email, get advice, and close requests quickly.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_right:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        with st.form("client_login", clear_on_submit=False):
            st.subheader("Enter your CLIENTNUM")
            st.markdown('<div class="login-note">Use your CLIENTNUM to continue.</div>', unsafe_allow_html=True)
            client_num_input = st.text_input("CLIENTNUM", placeholder="e.g., 710930508")
            submitted = st.form_submit_button("Continue")
        st.markdown("</div>", unsafe_allow_html=True)

        if submitted:
            if not client_num_input.strip().isdigit():
                st.error("Please enter a valid numeric CLIENTNUM.")
                return
            with st.spinner("Verifying your account..."):
                profile = fetch_customer_profile(int(client_num_input))
            if not profile:
                st.error("We could not find that CLIENTNUM. Please try again.")
                return
            st.session_state.client_num = int(client_num_input)
            st.session_state.client_profile = profile
            st.success("Account verified. Welcome back!")
            st.rerun()


def _render_chat(profile: Dict[str, str], logo_b64: Optional[str]) -> None:
    client_name = profile.get("name", "Client")

    header_left, header_right = st.columns([4, 1])
    with header_left:
        st.markdown(
            f"""
            <div class="chat-header">
                <div>
                    <div class="chat-title">Client Service Chat</div>
                    <div class="chat-sub">Hi {client_name}, ask anything about your account.</div>
                </div>
                <span class="chat-status">Online</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        st.button("Log out", on_click=_reset_session)

    st.markdown("<div class='chat-shell'>", unsafe_allow_html=True)
    if not st.session_state.chat_history:
        st.markdown(
            f"""
            <div class="chat-welcome">
                <h3>Hello {client_name}, how can we help?</h3>
                <p>Try one of these quick prompts.</p>
                <div>
                    <span class="pill">What is my current credit limit?</span>
                    <span class="pill">Change my email address</span>
                    <span class="pill">Increase my credit limit</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.write(content)

    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Type your message")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = send_chat_message(
                st.session_state.client_num,
                prompt,
                history=st.session_state.chat_history[:-1],
            )
        if response.get("ok"):
            assistant_text = response.get("data", {}).get(
                "message",
                "Thanks for reaching out. How can I help you further?",
            )
        else:
            error = response.get("error")
            if error == "chat_not_ready":
                assistant_text = (
                    "Our chatbot is being connected. Please check back soon or contact support."
                )
            else:
                assistant_text = (
                    "Sorry, the chat service is unavailable right now. Please try again later."
                )
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
        st.rerun()


def main() -> None:
    st.set_page_config(page_title="Serfy Bank Client Service", page_icon="💬", layout="centered")
    st.markdown(get_client_css(), unsafe_allow_html=True)

    _init_state()
    _render_hero(st.session_state.client_num)

    if not st.session_state.client_profile:
        _render_landing()
        return

    _render_chat(st.session_state.client_profile, _get_logo_base64())


if __name__ == "__main__":
    main()
