"""
Client-facing Streamlit styles (clean rebuild).
"""


def get_client_css() -> str:
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&family=Fraunces:wght@600;700&display=swap');

        :root {
            --bg: #0b0f14;
            --bg-2: #101720;
            --ink: #f8fafc;
            --muted: #b6c1ce;
            --card: #111827;
            --line: #1f2937;
            --accent: #f59e0b;
            --accent-2: #22d3ee;
        }

        html, body, [class*="stApp"] {
            background: radial-gradient(1200px 600px at 15% -15%, #1f2937 0%, var(--bg) 45%, #0b0f14 100%);
            color: var(--ink);
            font-family: "Manrope", system-ui, -apple-system, "Segoe UI", sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: "Fraunces", "Manrope", serif;
            color: var(--ink) !important;
            letter-spacing: 0.2px;
        }

        .main .block-container {
            max-width: 980px;
            padding: 2.5rem 2rem 7rem;
        }

        .hero {
            background: linear-gradient(140deg, #121826 0%, #0f172a 100%);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 2rem 2.25rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.75rem;
        }

        .hero-title {
            font-size: 2.2rem;
            margin: 0 0 0.35rem 0;
        }

        .hero-sub {
            color: var(--muted);
            font-size: 0.98rem;
        }

        .client-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #1f2a3a;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 0.75rem;
        }

        .landing-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .info-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.25);
            min-height: 120px;
        }

        .info-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
        }

        .info-card p {
            color: var(--muted);
            font-size: 0.92rem;
            margin: 0;
        }

        .login-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }

        .login-note {
            color: var(--muted);
            font-size: 0.92rem;
            margin-bottom: 0.6rem;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
            color: #f8fafc;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }

        .stButton > button:hover {
            filter: brightness(1.05);
        }

        .chat-header {
            background: #111827;
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }

        .chat-title {
            font-size: 1.1rem;
            font-weight: 700;
        }

        .chat-sub {
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 0.15rem;
        }

        .chat-status {
            background: #0f2f1f;
            color: #86efac;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
            border: 1px solid #14532d;
        }

        .chat-shell {
            background: #0f172a;
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
            margin-top: 1rem;
        }

        .chat-welcome {
            background: #111827;
            border: 1px solid #1f2937;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0 1.25rem;
        }

        .chat-welcome p {
            color: var(--muted);
        }

        .pill {
            display: inline-flex;
            align-items: center;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: #0f172a;
            color: #f8fafc;
            border: 1px solid #1f2a3a;
            margin-right: 0.4rem;
            font-size: 0.85rem;
        }

        /* Chat input dock */
        div[data-testid="stChatInput"] {
            position: fixed !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            bottom: 18px !important;
            width: min(760px, 92vw) !important;
            background: #0b0f16 !important;
            border-radius: 999px !important;
            border: 1px solid #1f2937 !important;
            padding: 6px 8px !important;
            box-shadow: 0 18px 36px rgba(0, 0, 0, 0.35) !important;
            z-index: 2000 !important;
        }

        div[data-testid="stChatInput"] textarea {
            color: #f8fafc !important;
        }

        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9aa4b2 !important;
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] div[data-testid="chatMessageContent"] {
            border-radius: 16px !important;
            padding: 0.75rem 1rem !important;
            background: #111827 !important;
            color: #e5e7eb !important;
            border: 1px solid #1f2937 !important;
        }

        div[data-testid="stChatMessage"][data-message-author="user"] div[data-testid="chatMessageContent"] {
            background: #111827 !important;
            color: #f9fafb !important;
            border: 1px solid #0b1220 !important;
        }

        div[data-testid="stChatMessage"][data-message-author="assistant"] div[data-testid="chatMessageContent"] {
            background: #0f172a !important;
            color: #e2e8f0 !important;
            border: 1px solid #1f2937 !important;
        }

        div[data-testid="stChatMessage"] div[data-testid="stMarkdown"] *,
        div[data-testid="stChatMessage"] [class^="st-emotion-cache-"] * {
            color: inherit !important;
        }

        div[data-testid="stChatMessage"][data-message-author="assistant"] div[data-testid="stMarkdown"] *,
        div[data-testid="stChatMessage"][data-message-author="assistant"] [class^="st-emotion-cache-"] * {
            color: #e2e8f0 !important;
        }

        div[data-testid="stChatMessage"][data-message-author="user"] div[data-testid="stMarkdown"] *,
        div[data-testid="stChatMessage"][data-message-author="user"] [class^="st-emotion-cache-"] * {
            color: #f9fafb !important;
        }

        @media (max-width: 900px) {
            .main .block-container { padding: 1.5rem 1.25rem 7rem; }
            .landing-grid { grid-template-columns: 1fr; }
            div[data-testid="stChatInput"] { width: 92vw !important; }
        }
    </style>
    """
