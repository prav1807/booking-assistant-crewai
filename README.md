# Flight Booking Assistant (Rasa Pro + CrewAI + Ollama)

An NLP-powered flight booking chatbot built with **Rasa Pro CALM**, **CrewAI**, **Ollama (llama3.1)**, and the **Duffel API** for real-time flight search.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Ollama LLM                           │
│                   (llama3.1 on GPU/CPU)                      │
│                     localhost:11434                          │
└──────────────┬───────────────────────────┬──────────────────┘
               │                           │
┌──────────────▼──────────────┐ ┌──────────▼──────────────────┐
│       Rasa Pro CALM         │ │   FastAPI Crew Service       │
│  (Conversational chatbot)   │ │  (5 CrewAI crews via HTTP)   │
│      localhost:5005         │ │      localhost:8000           │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │  POST /validate,search,...    │
┌──────────────▼──────────────┐                │
│   Action Server (THE SEAM)  │────────────────┘
│      localhost:5055         │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                       Duffel API                             │
│              (Real-time flight offers & booking)             │
└─────────────────────────────────────────────────────────────┘
```

- **Rasa Pro CALM** — Conversational AI with LLM-driven flow orchestration
- **CrewAI** — 5 role-specialised agent crews for validation, search, policy, review, and safety
- **Ollama (llama3.1:latest)** — Local LLM (configurable via `OLLAMA_MODEL` env var)
- **Duffel API** — Real-time flight offers and booking
- **Supabase** — Backend data storage

## Prerequisites

- Python 3.13 (Rasa Pro does not support 3.14+)
- [Ollama](https://ollama.com) with `llama3.1` model
- NVIDIA GPU with updated drivers (or CPU fallback)
- Rasa Pro license key
- Duffel API access token

## Setup

### 1. Create virtual environment

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install rasa-pro --extra-index-url=https://europe-west3-python.pkg.dev/rasa-releases/rasa-pro-python/simple/
pip install crewai crewai-tools dateparser python-dotenv
pip install fastapi uvicorn requests
pip install qdrant-client sentence-transformers pandas
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
DUFFEL_ACCESS_TOKEN=your_duffel_token
DUFFEL_BASE_URL=https://api.duffel.com
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
RASA_PRO_LICENSE=your_rasa_pro_license
CREW_SERVICE_URL=http://localhost:8000
OLLAMA_MODEL=llama3.1:latest
```

### 4. Pull the Ollama model

```bash
ollama pull llama3.1
```

### 5. Fix WebSocket for Rasa Inspector (required)

```bash
pip install "websockets==11.0.3" --no-deps
```

## Running

### Full System (Rasa + CrewAI integrated)

Start all four services:

```bash
# Terminal 1 — Ollama (GPU)
ollama serve

# Terminal 2 — CrewAI Crew Service (FastAPI on port 8000)
python -X utf8 -m uvicorn crewai_agents.crew_service:app --host 0.0.0.0 --port 8000

# Terminal 3 — Rasa Action Server (port 5055)
cd booking-assistant
python -m rasa_sdk --actions actions -p 5055

# Terminal 4 — Rasa Inspector UI (port 5005)
cd booking-assistant
set RASA_PRO_LICENSE=your_license_key
rasa inspect
```

Then open the Inspector UI: **http://localhost:5005/webhooks/inspector/inspect.html**

### API-only mode (no UI)

```bash
# Replace "rasa inspect" with:
rasa run --enable-api --cors "*"
```

Send messages via REST:

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook ^
  -H "Content-Type: application/json" ^
  -d "{\"sender\": \"user\", \"message\": \"I want to book a flight\"}"
```

### Standalone CrewAI Agent (no Rasa)

```bash
# Make sure Ollama is running first
ollama serve

# Run the agent directly
python -X utf8 -m crewai_agents.main "Return flights from London to New York, departing 1 October 2026, returning 15 October, business class, 2 passengers"
```

## Project Structure

```
booking-assistant-crewai/
├── .env                              # Environment variables (git-ignored)
├── .gitignore
├── README.md
├── booking-assistant/                # Rasa Pro chatbot
│   ├── config.yml                    # Rasa pipeline config
│   ├── domain.yml                    # Slots, responses, actions
│   ├── endpoints.yml                 # Ollama + action endpoint config
│   ├── actions/
│   │   ├── __init__.py
│   │   ├── actions.py                # Custom actions (date validation, search)
│   │   ├── crew_actions.py           # THE SEAM — actions that call crew service
│   │   ├── duffel_client.py          # Duffel API client
│   │   └── tests/
│   │       └── test_dates.py
│   └── data/
│       └── flows/
│           ├── search_flights.yml
│           └── available_bookings.yml
└── crewai_agents/                    # CrewAI multi-agent layer
    ├── __init__.py
    ├── llm_config.py                 # Centralised LLM config (change model once)
    ├── crew_service.py               # FastAPI service (port 8000, 5 endpoints)
    ├── main.py                       # CLI entry point (flight search)
    ├── flight_search_crew.py         # Flight search agent + crew (Roles 2+3)
    ├── validation_crew.py            # Booking request validation crew (Role 1)
    ├── policy_crew.py                # Policy & visa eligibility crew (Role 4)
    ├── booking_review_crew.py        # Pre-booking payload verification (Role 5)
    ├── guardrail_crew.py             # Outbound message safety screening (Role 6)
    ├── tools/
    │   ├── __init__.py
    │   ├── flight_tools.py           # FlightSearchTool + AirportLookupTool
    │   ├── validation_tools.py       # Date, route, cabin, passenger validators
    │   ├── policy_tools.py           # PolicyRetrievalTool + VisaEligibilityTool
    │   ├── booking_review_tools.py   # PayloadVerificationTool
    │   └── guardrail_tools.py        # RegexPreFilterTool + FactualClaimDetector
    ├── data/
    │   ├── policy_corpus.json        # Policy knowledge base (IATA, EU, US regs)
    │   └── passport-index-tidy.csv   # Visa eligibility dataset (39,601 pairs)
    └── tests/
        ├── test_e2e.py               # Full end-to-end test (all 5 endpoints)
        ├── test_edge_cases.py        # Flight search tool tests (13 tests)
        ├── test_validation_crew.py   # Validation crew tests
        └── test_policy_crew.py       # Policy & visa crew tests
```

## Conversation Flows (Rasa)

1. **search_flights** — Collects trip type, origin, destination, dates, passengers, cabin class → searches Duffel for flights
2. **available_bookings** — Presents flight offers and lets the user select one

## Rasa ↔ CrewAI Integration

The integration follows the report's "seam" pattern (Section 4.4):

```
Rasa CALM Flow                    Crew Service (FastAPI :8000)
─────────────                    ────────────────────────────
collect slots
       │
       ▼
action_validate_booking ──POST /validate──▶ ValidationCrew (Role 1)
       │                                         │
       ◀──── {valid, errors, duffel_request} ────┘
       │
  branch on valid
       │
action_search_flights ──────────────────▶ Duffel API (direct)
action_crew_search ─────POST /search────▶ FlightSearchCrew (Roles 2+3)
       │
action_answer_policy ───POST /policy────▶ PolicyCrew (Role 4)
       │
action_verify_booking ──POST /verify────▶ BookingReviewCrew (Role 5)
       │
action_guardrail_check ─POST /guardrail─▶ GuardrailCrew (Role 6)
```

**Key property:** Crews return structured data only — all user-facing text is rendered by CALM flow responses. This preserves the constrained-generation guarantee.

### Crew Service Endpoints

| Endpoint | Crew | Role | Purpose |
|----------|------|------|---------|
| `POST /validate` | ValidationCrew | 1 | Validate slots, build Duffel request |
| `POST /search` | FlightSearchCrew | 2+3 | Search and rank flights |
| `POST /policy` | PolicyCrew | 4 | Answer policy/visa questions |
| `POST /verify` | BookingReviewCrew | 5 | Pre-booking payload verification |
| `POST /guardrail` | GuardrailCrew | 6 | Outbound message safety screening |
| `GET /health` | — | — | Health check (reports 5 crews) |

## CrewAI Crews (5/5 implemented)

### 1. ValidationCrew (Role 1 — Validate & Build Request)
Validates all booking slots and builds the Duffel API request payload:
- **Deterministic pre-check** — instant tool-level validation (no LLM needed)
- **Route validation** — IATA code format, origin ≠ destination
- **Date validation** — real dates, not in the past, within booking horizon
- **Consistency checks** — return trips need return dates, cabin class normalization
- **Structured verdict** — returns `{valid, errors, warnings, duffel_request}`

### 2. FlightSearchCrew (Roles 2+3 — Search & Rank)
Searches and compares flights via the Duffel API:
- **One-way & round-trip** flights
- **City name resolution** — automatically looks up IATA codes (e.g. "Mauritius" → MRU)
- **Multiple cabin classes** — economy, premium_economy, business, first
- **Multi-passenger** bookings (up to 9 adults)
- **Vague requests** — interprets "next month", "cheapest", etc.
- **Graceful error handling** — invalid routes, past dates, nonexistent airports

```bash
python -X utf8 -m crewai_agents.main "Flights from Mauritius to Dubai on 20 Sep 2026, economy, 1 passenger"
```

### 3. PolicyCrew (Role 4 — Grounded Policy & Visa Answers)
Answers policy and visa eligibility questions from grounded sources:
- **PolicyRetrievalAgent** — RAG over Qdrant-indexed policy corpus (IATA baggage, dangerous goods, EU 1107/2006, US 14 CFR 382, fare rules) using BGE-small-en-v1.5 embeddings
- **EligibilityAgent** — structured visa lookup from passport-index dataset (39,601 passport-destination pairs)
- **Grounded only** — every answer cites its source; never fabricates policy information

### 4. BookingReviewCrew (Role 5 — Pre-Booking Verification)
Verifies the assembled order payload before submission to the airline:
- **Passenger field checks** — all required fields present (name, DOB, title, gender, email, phone)
- **Cross-referencing** — passenger details match conversation slots
- **Offer validation** — correct offer ID, not expired
- **Route consistency** — order origin/destination matches slot values
- **Passenger count** — payload matches requested count
- **Deterministic-first** — no LLM for clear-cut mismatches; LLM only for ambiguous cases

### 5. GuardrailCrew (Role 6 — Outbound Message Safety)
Screens every outbound message with a tiered design:
- **Tier 1 (regex, all messages)** — fast PII detection (credit cards, passports, emails, phones, SSNs) and prohibited phrasing (absolute guarantees, legal promises, unhedged price commitments)
- **Tier 2 (LLM, factual claims only)** — invoked only when message contains price/policy/regulatory claims, checking for over-commitment and ungrounded statements
- **Fail-open** — if LLM is unavailable, message passes (regex tier still protects)
- **Suppression** — rejected messages are blocked, not sent with a warning

## Testing

```bash
# Full end-to-end test (all 5 crew service endpoints)
# Requires: Ollama + crew service running
python -X utf8 crewai_agents\tests\test_e2e.py

# Flight search tool edge cases (13 tests)
python -X utf8 crewai_agents\tests\test_edge_cases.py

# Validation crew (valid + invalid request tests)
python -X utf8 -m crewai_agents.tests.test_validation_crew

# Policy & visa crew (baggage policy + visa eligibility tests)
python -X utf8 -m crewai_agents.tests.test_policy_crew
```

## Notes

- **Python 3.13 required** — Rasa Pro doesn't support 3.14+. Use a `.venv` with Python 3.13.
- **WebSocket fix** — `websockets==11.0.3` is required for Rasa Inspector. Newer versions break the Sanic WebSocket handshake.
- **GPU acceleration** — Ollama uses NVIDIA GPU by default. For CPU fallback: `set CUDA_VISIBLE_DEVICES=-1` and `set OLLAMA_LLM_LIBRARY=cpu`.
- **4 services** — Full system requires Ollama + Crew Service + Action Server + Rasa. See "Running" section.
- **Centralised LLM config** — Change model in `.env` via `OLLAMA_MODEL=qwen3:8b` (or edit `crewai_agents/llm_config.py`).
- **Crew timeout** — Default 120s per crew call. Configurable via `CREW_SERVICE_TIMEOUT` env var.
- **CrewAI performance** — ~30s per crew call on GPU vs ~3-5min on CPU.
- **All 5 crews implemented** — ValidationCrew, FlightSearchCrew, PolicyCrew, BookingReviewCrew, GuardrailCrew.
