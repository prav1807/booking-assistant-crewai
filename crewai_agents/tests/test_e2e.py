"""Full end-to-end test of all 5 crew service endpoints."""
import sys
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}")
        if detail:
            print(f"         {detail[:200]}")


def section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def post(endpoint, payload, timeout=300):
    try:
        r = requests.post(f"{BASE}{endpoint}", json=payload, timeout=timeout)
        return r.status_code, r.json()
    except Exception as e:
        return 0, {"error": str(e)}


# =====================================================================
section("0. Health Check")
# =====================================================================
code, data = requests.get(f"{BASE}/health").status_code, requests.get(f"{BASE}/health").json()
test("Health endpoint returns OK", code == 200 and data["status"] == "ok")
test("Reports 5 crews", data.get("crews") == 5, f"Got: {data}")


# =====================================================================
section("1. ValidationCrew (Role 1) — POST /validate")
# =====================================================================

# Valid request
code, data = post("/validate", {
    "origin_code": "MRU", "destination_code": "DXB",
    "departure_date": "2026-10-01", "passenger_count": 1,
    "cabin_class": "economy", "trip_type": "one_way"
})
test("Valid booking → valid=true", data.get("valid") == True, json.dumps(data))
test("Valid booking → no errors", data.get("errors") == [])
test("Valid booking → duffel_request present", data.get("duffel_request") is not None)

# Invalid request
code, data = post("/validate", {
    "origin_code": "MRU", "destination_code": "MRU",
    "departure_date": "2020-01-01", "passenger_count": 0,
    "cabin_class": "luxury", "trip_type": "return"
})
test("Invalid booking → valid=false", data.get("valid") == False)
test("Invalid booking → has errors", len(data.get("errors", [])) >= 3, json.dumps(data.get("errors")))
test("Invalid booking → no duffel_request", data.get("duffel_request") is None)


# =====================================================================
section("2. FlightSearchCrew (Roles 2+3) — POST /search")
# =====================================================================

code, data = post("/search", {
    "user_request": "One-way flights from MRU to DXB, departing 2026-10-01, economy, 1 passenger"
})
test("Flight search → success=true", data.get("success") == True, data.get("error", ""))
test("Flight search → result has content", len(data.get("result", "")) > 50)


# =====================================================================
section("3. PolicyCrew (Role 4) — POST /policy")
# =====================================================================

# Baggage policy question
code, data = post("/policy", {
    "question": "What is the carry-on baggage weight limit?"
})
test("Policy (baggage) → answered=true", data.get("answered") == True)
test("Policy (baggage) → mentions kg", "kg" in data.get("answer", "").lower() or "7" in data.get("answer", ""))

# Visa eligibility
code, data = post("/policy", {
    "question": "Do I need a visa for Japan?",
    "passport_country": "Mauritius",
    "destination_country": "Japan"
})
test("Visa (MRU→JPN) → answered=true", data.get("answered") == True)
test("Visa (MRU→JPN) → visa-free", "free" in data.get("answer", "").lower())
test("Visa (MRU→JPN) → type=eligibility", data.get("question_type") == "eligibility")


# =====================================================================
section("4. BookingReviewCrew (Role 5) — POST /verify")
# =====================================================================

# Valid payload
code, data = post("/verify", {
    "order_payload": {"data": {"slices": [{"origin": "MRU", "destination": "DXB"}], "passengers": [{"type": "adult"}]}},
    "passenger_details": {"given_name": "John", "family_name": "Doe", "born_on": "1990-01-01", "title": "Mr", "gender": "male", "email": "j@t.com", "phone": "+230123"},
    "selected_offer": {"id": "off_1", "expires_at": "2027-01-01T00:00:00Z"},
    "conversation_slots": {"origin_code": "MRU", "destination_code": "DXB", "selected_offer_id": "off_1", "passenger_count": 1, "given_name": "John", "family_name": "Doe", "born_on": "1990-01-01", "passenger_title": "Mr", "passenger_gender": "male", "passenger_email": "j@t.com", "passenger_phone": "+230123"}
}, timeout=30)
test("Verify (valid) → verified=true", data.get("verified") == True)
test("Verify (valid) → no issues", data.get("issues") == [])

# Mismatched payload
code, data = post("/verify", {
    "order_payload": {"data": {"slices": [{"origin": "LHR", "destination": "DXB"}], "passengers": [{"type": "adult"}]}},
    "passenger_details": {"given_name": "John", "family_name": "", "born_on": "1990-01-01", "title": "Mr", "gender": "male", "email": "j@t.com", "phone": ""},
    "selected_offer": {"id": "off_2", "expires_at": "2020-01-01T00:00:00Z"},
    "conversation_slots": {"origin_code": "MRU", "destination_code": "DXB", "selected_offer_id": "off_1", "passenger_count": 2}
}, timeout=30)
test("Verify (mismatch) → verified=false", data.get("verified") == False)
test("Verify (mismatch) → catches issues", len(data.get("issues", [])) >= 4, str(data.get("issues")))


# =====================================================================
section("5. GuardrailCrew (Role 6) — POST /guardrail")
# =====================================================================

# Safe message
code, data = post("/guardrail", {
    "draft_message": "Here are 3 flight options from Mauritius to Dubai."
}, timeout=30)
test("Guardrail (safe) → passed=true", data.get("passed") == True)
test("Guardrail (safe) → tier=regex", data.get("tier") == "regex")

# PII blocked
code, data = post("/guardrail", {
    "draft_message": "Your card 4111-1111-1111-1111 has been charged GBP 205."
}, timeout=30)
test("Guardrail (PII) → passed=false", data.get("passed") == False)
test("Guardrail (PII) → tier=regex", data.get("tier") == "regex")
test("Guardrail (PII) → reason mentions credit_card", "credit_card" in data.get("reason", ""))

# Prohibited phrasing
code, data = post("/guardrail", {
    "draft_message": "We guarantee this price will never change and the seat is confirmed forever."
}, timeout=30)
test("Guardrail (prohibited) → passed=false", data.get("passed") == False)

# Message with factual claims (triggers LLM review)
code, data = post("/guardrail", {
    "draft_message": "Under EU Regulation 1107/2006, you are entitled to free wheelchair assistance at the airport."
}, timeout=120)
test("Guardrail (factual claim) → returns a result", "passed" in data)
test("Guardrail (factual claim) → tier is llm", data.get("tier") == "llm", f"tier={data.get('tier')}")


# =====================================================================
section("SUMMARY")
# =====================================================================
total = PASS + FAIL
print(f"\n  Results: {PASS}/{total} passed, {FAIL}/{total} failed")
if FAIL == 0:
    print("  ALL TESTS PASSED!")
print()
