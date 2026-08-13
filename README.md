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
│       Rasa Pro CALM         │ │     CrewAI Flight Agent      │
│  (Conversational chatbot)   │ │  (Autonomous flight search)  │
│      localhost:5005         │ │     Standalone CLI/API        │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │
┌──────────────▼──────────────┐                │
│      Action Server          │                │
│      localhost:5055         │                │
└──────────────┬──────────────┘                │
               │                               │
┌──────────────▼───────────────────────────────▼──────────────┐
│                       Duffel API                             │
│              (Real-time flight offers & booking)             │
└─────────────────────────────────────────────────────────────┘
```

- **Rasa Pro CALM** — Conversational AI with LLM-driven flow orchestration
- **CrewAI** — Autonomous agent that searches, compares, and recommends flights
- **Ollama (llama3.1:latest)** — Local LLM running on RTX 3070 GPU
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
```

### 3. Configure environment

Create a `.env` file in the project root:

```env
DUFFEL_ACCESS_TOKEN=your_duffel_token
DUFFEL_BASE_URL=https://api.duffel.com
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_supabase_key
RASA_PRO_LICENSE=your_rasa_pro_license
```

### 4. Pull the Ollama model

```bash
ollama pull llama3.1
```

## Running

### Option A: Rasa Chatbot (Conversational Flow)

Start all three services:

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — Action Server
cd booking-assistant
python -m rasa_sdk --actions actions -p 5055

# Terminal 3 — Rasa Server
cd booking-assistant
set RASA_PRO_LICENSE=your_license_key
rasa train
rasa run --enable-api --cors "*"
```

Send messages via the REST webhook:

```bash
curl -X POST http://localhost:5005/webhooks/rest/webhook ^
  -H "Content-Type: application/json" ^
  -d "{\"sender\": \"user\", \"message\": \"I want to book a flight\"}"
```

### Option B: CrewAI Agent (Autonomous Flight Search)

```bash
# Make sure Ollama is running first
ollama serve

# Run the agent with a natural language request
python -X utf8 -m crewai_agents.main "Return flights from London to New York, departing 1 October 2026, returning 15 October, business class, 2 passengers"
```

The CrewAI agent will autonomously:
1. Look up IATA airport codes from city names
2. Search the Duffel API for available flights
3. Analyse and rank results by price, duration, and stops
4. Present a recommendation

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
│   │   ├── duffel_client.py          # Duffel API client
│   │   └── tests/
│   │       └── test_dates.py
│   └── data/
│       └── flows/
│           ├── search_flights.yml
│           └── available_bookings.yml
└── crewai_agents/                    # CrewAI autonomous agent
    ├── __init__.py
    ├── main.py                       # CLI entry point
    ├── flight_search_crew.py         # Agent, task, crew definitions
    ├── tools/
    │   ├── __init__.py
    │   └── flight_tools.py           # FlightSearchTool + AirportLookupTool
    └── tests/
        └── test_edge_cases.py        # Edge case test suite (13 tests)
```

## Conversation Flows (Rasa)

1. **search_flights** — Collects trip type, origin, destination, dates, passengers, cabin class → searches Duffel for flights
2. **available_bookings** — Presents flight offers and lets the user select one

## CrewAI Agent Capabilities

The flight search agent handles:
- **One-way & round-trip** flights
- **City name resolution** — automatically looks up IATA codes (e.g. "Mauritius" → MRU)
- **Multiple cabin classes** — economy, premium_economy, business, first
- **Multi-passenger** bookings (up to 9 adults)
- **Vague requests** — interprets "next month", "cheapest", etc.
- **Graceful error handling** — invalid routes, past dates, nonexistent airports

## Testing

```bash
# Run edge case tests for the CrewAI tools
python -X utf8 crewai_agents\tests\test_edge_cases.py
```

## Notes

- Rasa Pro requires Python ≤3.13. A `.venv` with Python 3.13 is used for this project.
- Ollama uses the RTX 3070 GPU by default. For CPU fallback: `set CUDA_VISIBLE_DEVICES=-1` and `set OLLAMA_LLM_LIBRARY=cpu`.
- The Rasa action server registers 9 custom actions for place resolution, date validation, and flight search.
- CrewAI agent responses are faster on GPU (~30s) vs CPU (~3-5min per request).

