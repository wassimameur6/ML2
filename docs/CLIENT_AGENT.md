# Client Agent (Serfy Bank) - How It Works

This document explains the client-facing chat agent: the technologies used, how requests flow through the system, and the business logic for key actions (email change, credit-limit increase).

## 1) Technology stack

**Runtime / UI**
- **Streamlit** (`webapp/client_app.py`): client UI with a landing page and chat view.
- **Requests** (`webapp/client_api.py`): calls the API from Streamlit.

**API**
- **FastAPI + Uvicorn** (`serving/api.py`): exposes `/client/chat`.
- **Pydantic**: request/response validation for the chat endpoint.

**Agent Core**
- **OpenRouter** (OpenAI-compatible) via `httpx` (`agent/openrouter_client.py`).
- **Tool orchestration** (`agent/client_chat_agent.py`): runs tools and formats deterministic responses.

**Data**
- **Pandas** CSV store (`data/churn2.csv`) via `agent/customer_repository.py`.

**Email**
- **SMTP** (`agent/churn_agent.py` -> `EmailService`): sends verification codes for email changes.

---

## 2) High-level flow

1. **User opens UI** (`webapp/client_app.py`).
2. **Landing page** asks for `CLIENTNUM`.
3. The UI calls `GET /customers/{CLIENTNUM}` to fetch profile info.
4. User enters a chat message -> UI calls:
   - `POST /client/chat` with `client_num`, `message`, and `history`.
5. `serving/api.py` forwards to the agent:
   - `ClientChatAgent.reply(...)`.
6. Agent either:
   - Calls LLM tools and uses their responses, or
   - Falls back to deterministic tool logic for credit-limit requests if the model does not call a tool.

### Request flow diagram (agent calls)

```
Client UI (Streamlit)
  |  POST /client/chat (message + history)
  v
FastAPI (serving/api.py)
  |  ClientChatAgent.reply(...)
  v
OpenRouter LLM (tool-aware)
  |-- tool: get_user_info ----------> CustomerRepository -> churn2.csv
  |-- tool: change_email -----------> EmailService (SMTP) -> send code
  |-- tool: validate_change_email --> CustomerRepository -> churn2.csv
  |-- tool: increase_credit_limit -> CustomerRepository -> churn2.csv
  |
  v
Final assistant reply (based on tool result)
```

---

## 3) Agent logic (core)

**File:** `agent/client_chat_agent.py`

### 3.1 Message handling
- Incoming message + conversation history are sent to the LLM.
- System prompt forces a tool-first approach for sensitive actions.
- If the model does not call tools for credit-limit changes, the agent **directly processes the request** using its own logic (no hallucinated replies).

### 3.2 Tools exposed to the LLM

1) **`get_user_info`**
- Returns a profile subset (name, email, card tier, credit limit, etc.).
- Reads from CSV through `CustomerRepository`.

2) **`change_email`**
- Sends a verification code to a **new** email address.
- Uses SMTP via `EmailService`.
- Stores a **pending change** in memory (per client).

3) **`validate_change_email`**
- Validates the code sent to the new email.
- If valid, updates `Email` in `data/churn2.csv`.

4) **`increase_credit_limit`**
- Accepts **percentage OR amount**.
- Enforces card tier limits:
  - **Gold:** up to +20%
  - **Platinum/Platinium:** up to +50%
  - Other tiers: **not eligible**
- Returns a structured result: `approved` or `refused` plus reason.
- On approval, updates `Credit_Limit` in `data/churn2.csv`.

---

## 4) Deterministic responses (no hallucinations)

To prevent incorrect confirmations (e.g., "limit increased" when it was not), the agent:

- **Formats the response directly** from the tool result:
  - Approved -> new limit printed with correct formatting.
  - Refused -> reason printed ("category not eligible" or "above limit").
- This avoids weird output such as `77,661.A70...` and prevents the model from inventing a success.

---

## 5) Credit-limit increase rules

**Input:** either
- `percent` (e.g., `10%`)
- `amount` (e.g., `$2000`)

**Decision logic**
- If card tier not eligible -> **refuse** with reason `category_not_eligible`
- If request > tier limit -> **refuse** with reason `above_limit`
- Otherwise -> **approve** and update CSV

---

## 6) Email change flow

1. User asks to change email.
2. Agent runs `change_email`:
   - Generates a 6-digit code.
   - Sends it to the new email using SMTP.
3. User provides the code.
4. Agent runs `validate_change_email`:
   - If valid -> updates `Email` in CSV.
   - If expired/invalid -> refuses with a clear reason.

---

## 7) Data persistence

Changes are stored **directly in `data/churn2.csv`**:
- `Email` updated after verification.
- `Credit_Limit` updated after approved increase.

`CustomerRepository` handles read/write via pandas.

---

## 8) Configuration

**Required for chat:**
- `OPENROUTER_API_KEY`

**Required for email verification:**
- `SMTP_HOST`
- `SMTP_PORT`
- `SMTP_USER`
- `SMTP_PASSWORD`
- `SENDER_EMAIL`

Optional:
- `COMPANY_NAME`
- `API_BASE_URL`

---

## 9) Key files

- `webapp/client_app.py` - Streamlit client UI.
- `webapp/client_api.py` - API helper functions.
- `agent/client_chat_agent.py` - tool orchestration and logic.
- `agent/openrouter_client.py` - OpenRouter HTTP client.
- `agent/customer_repository.py` - CSV reader/writer.
- `serving/api.py` - FastAPI endpoint `/client/chat`.
- `data/churn2.csv` - source of truth for client data.

---

## 10) How to run the client agent

### Local (no Docker)

**Terminal 1 - API**
```
cd /Users/oujghou/Desktop/ML2/serving
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8080
```

**Terminal 2 - Client UI**
```
cd /Users/oujghou/Desktop/ML2/webapp
pip install -r requirements.txt
streamlit run client_app.py
```

If your API runs elsewhere:
```
export API_URL="http://localhost:8080"
```

### Docker (client agent + API)

From the project root:
```
docker compose up --build serving-api client-webapp
```

Then open:
- Client UI: `http://localhost:8502`
- API: `http://localhost:8080`

If you want the admin UI too:
```
docker compose up --build serving-api webapp client-webapp
```
