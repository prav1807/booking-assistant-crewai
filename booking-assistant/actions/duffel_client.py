import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

import re
from datetime import datetime, timezone

import re as _re

load_dotenv()

logger = logging.getLogger(__name__)

DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"
REQUEST_TIMEOUT_SECONDS = 10
MAX_OFFERS_SHOWN = 5

# Development-time cache. Replaced by Redis once the integration is stable.
_place_cache: Dict[str, List[Dict[str, Any]]] = {}


class DuffelError(RuntimeError):
    """Raised when Duffel cannot be reached or returns an unusable response."""


@dataclass
class PlaceResolution:
    """Outcome of resolving a free-text place name to a bookable code.

    status is one of: resolved, ambiguous, unknown, error
    """

    status: str
    code: Optional[str] = None
    display: Optional[str] = None
    options: List[Dict[str, str]] = field(default_factory=list)


def _headers() -> Dict[str, str]:
    token = os.getenv("DUFFEL_ACCESS_TOKEN")
    if not token:
        raise DuffelError("DUFFEL_ACCESS_TOKEN is not set — check your .env file.")
    return {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }


def suggest_places(query: str) -> List[Dict[str, Any]]:
    """Call Duffel's place suggestions endpoint, with a simple cache."""
    key = (query or "").strip().lower()
    if not key:
        return []

    if key in _place_cache:
        logger.info("Place cache HIT for %r", key)
        return _place_cache[key]

    logger.info("Place cache MISS for %r — calling Duffel", key)
    try:
        response = requests.get(
            f"{DUFFEL_API_BASE}/places/suggestions",
            params={"query": key},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DuffelError(f"Could not reach Duffel: {exc}") from exc

    if response.status_code != 200:
        raise DuffelError(
            f"Duffel returned {response.status_code}: {response.text[:200]}"
        )

    places = response.json().get("data", [])
    _place_cache[key] = places
    return places


def _airport_option(airport: Dict[str, Any]) -> Dict[str, str]:
    return {
        "iata_code": airport.get("iata_code", ""),
        "name": airport.get("name", ""),
        "city_name": airport.get("city_name", "") or "",
    }


def resolve_place(query: str) -> PlaceResolution:
    """Resolve free text to a single airport, a choice of airports, or nothing.

    A city with more than one airport is deliberately NOT resolved
    automatically — the user is asked which airport they mean (FR2).
    """
    try:
        places = suggest_places(query)
    except DuffelError as exc:
        logger.error("Place lookup failed for %r: %s", query, exc)
        return PlaceResolution(status="error")

    if not places:
        return PlaceResolution(status="unknown")

    top = places[0]
    place_type = top.get("type")

    if place_type == "airport":
        code = top.get("iata_code")
        if not code:
            return PlaceResolution(status="unknown")
        city = top.get("city_name") or ""
        display = f"{top.get('name', code)} ({code})"
        if city:
            display = f"{city} — {display}"
        return PlaceResolution(status="resolved", code=code, display=display)

    if place_type == "city":
        airports = [
            a
            for a in top.get("airports", [])
            if a.get("iata_code") and _is_commercial(a)
        ]
        if len(airports) == 1:
            only = airports[0]
            code = only["iata_code"]
            return PlaceResolution(
                status="resolved",
                code=code,
                display=f"{top.get('name', '')} — {only.get('name', code)} ({code})",
            )

        if len(airports) > 1:
            return PlaceResolution(
                status="ambiguous",
                display=top.get("name", query),
                options=[_airport_option(a) for a in airports],
            )

    return PlaceResolution(status="unknown")


def format_options(city: str, options: List[Dict[str, str]]) -> str:
    lines = [f"{opt['iata_code']} — {opt['name']}" for opt in options]
    return f"{city} has several airports:\n" + "\n".join(lines)


def match_choice(
    choice: str, options: List[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Match a user's free-text choice against the offered airports."""
    if not choice:
        return None
    needle = choice.strip().lower()

    for opt in options:
        if needle == opt["iata_code"].lower():
            return opt
    for opt in options:
        if needle == opt["name"].lower():
            return opt
    matches = [o for o in options if needle in o["name"].lower()]
    if len(matches) == 1:
        return matches[0]
    return None


# Airports without scheduled commercial service. Duffel's suggestions endpoint
# returns these alongside commercial airports, and offering them to a user
# leads to searches that can never return offers.
# LIMITATION: keyword matching is heuristic; a production system would use a
# curated list of commercially-served airports.
NON_COMMERCIAL_MARKERS = (
    "seaplane base",
    "air base",
    "airbase",
    "air force base",
    "heliport",
    "airfield",
    "biggin hill",
)


def _is_commercial(airport: Dict[str, Any]) -> bool:
    name = (airport.get("name") or "").lower()
    return not any(marker in name for marker in NON_COMMERCIAL_MARKERS)


# ---------------------------------------------------------------------------
# Offer requests
# ---------------------------------------------------------------------------

# Duffel caps how long it will wait for airline responses.
SEARCH_TIMEOUT_SECONDS = 60


def build_slices(
    origin_code: str,
    destination_code: str,
    departure_date: str,
    return_date: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Build the slice list. Trip type is expressed by the number of slices."""
    slices = [
        {
            "origin": origin_code,
            "destination": destination_code,
            "departure_date": departure_date,
        }
    ]
    if return_date:
        slices.append(
            {
                "origin": destination_code,
                "destination": origin_code,
                "departure_date": return_date,
            }
        )
    return slices


def create_offer_request(
    slices: List[Dict[str, str]],
    passenger_count: int,
    cabin_class: str,
) -> List[Dict[str, Any]]:
    """Create an offer request and return the offers it produced.

    MVP LIMITATION: every passenger is sent as an adult. Child and
    infant fares are out of scope (see Chapter 7).
    """
    payload = {
        "data": {
            "slices": slices,
            "passengers": [{"type": "adult"} for _ in range(passenger_count)],
            "cabin_class": cabin_class,
        }
    }

    logger.info("OFFER REQUEST | payload=%s", payload)

    try:
        response = requests.post(
            f"{DUFFEL_API_BASE}/air/offer_requests",
            params={"return_offers": "true"},
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DuffelError(f"Could not reach Duffel: {exc}") from exc

    if response.status_code not in (200, 201):
        logger.error(
            "OFFER REQUEST FAILED | %s | %s", response.status_code, response.text[:500]
        )
        raise DuffelError(
            f"Duffel returned {response.status_code}: {response.text[:200]}"
        )

    return response.json().get("data", {}).get("offers", [])


def summarise_offer(offer: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a Duffel offer to the fields needed for display.

    Every value here comes from the provider. Nothing is generated.
    """
    slices = offer.get("slices", [])
    legs = []

    for sl in slices:
        segments = sl.get("segments", [])
        if not segments:
            continue
        first, last = segments[0], segments[-1]
        legs.append(
            {
                "origin": (first.get("origin") or {}).get("iata_code", ""),
                "destination": (last.get("destination") or {}).get("iata_code", ""),
                "departing_at": first.get("departing_at", ""),
                "arriving_at": last.get("arriving_at", ""),
                "stops": max(len(segments) - 1, 0),
                "duration": sl.get("duration", ""),
            }
        )

    owner = offer.get("owner") or {}

    return {
        "id": offer.get("id", ""),
        "airline": owner.get("name", ""),
        "total_amount": offer.get("total_amount", ""),
        "total_currency": offer.get("total_currency", ""),
        "expires_at": offer.get("expires_at", ""),
        "legs": legs,
        "passenger_ids": [
            p.get("id") for p in offer.get("passengers", []) if p.get("id")
        ],
    }


_DURATION_RE = re.compile(
    r"P(?:(?P<days>\d+)D)?T?(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
)


def format_duration(iso_duration: str) -> str:
    """Convert an ISO 8601 duration (P1DT4H30M) into '28h 30m'."""
    if not iso_duration:
        return ""
    match = _DURATION_RE.match(iso_duration)
    if not match:
        return ""
    days = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    total_hours = days * 24 + hours
    return f"{total_hours}h {minutes:02d}m"


def format_time(iso_timestamp: str) -> str:
    """Render a Duffel local departure/arrival time as '15 Sep 15:20'."""
    if not iso_timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
    except ValueError:
        return iso_timestamp
    return dt.strftime("%d %b %H:%M")


def format_leg(leg: Dict[str, Any]) -> str:
    stops = leg.get("stops", 0)
    stop_text = (
        "direct" if stops == 0 else f"{stops} stop" if stops == 1 else f"{stops} stops"
    )
    return (
        f"{leg['origin']} → {leg['destination']}  "
        f"{format_time(leg['departing_at'])} – {format_time(leg['arriving_at'])}  "
        f"({format_duration(leg['duration'])}, {stop_text})"
    )


def format_offer(index: int, offer: Dict[str, Any]) -> str:
    """Render one offer for display. Every value comes from the provider."""
    lines = [
        f"{index}. {offer['airline']} — "
        f"{offer['total_currency']} {offer['total_amount']}"
    ]
    for leg in offer.get("legs", []):
        lines.append(f"   {format_leg(leg)}")
    return "\n".join(lines)


def format_offer_list(offers: List[Dict[str, Any]]) -> str:
    return "\n\n".join(
        format_offer(i, o) for i, o in enumerate(offers[:MAX_OFFERS_SHOWN], start=1)
    )


def minutes_until_expiry(offer: Dict[str, Any]) -> Optional[int]:
    """Minutes remaining before this offer can no longer be booked."""
    raw = offer.get("expires_at")
    if not raw:
        return None
    try:
        expires = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    delta = expires - datetime.now(timezone.utc)
    return int(delta.total_seconds() // 60)


EMAIL_RE = _re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", _re.I)
E164_RE = _re.compile(r"^\+[1-9]\d{7,14}$")

VALID_TITLES = ("mr", "ms", "mrs", "miss", "dr")
VALID_GENDERS = ("m", "f")


def normalise_title(raw: str) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip().lower().rstrip(".")
    return text if text in VALID_TITLES else None


def normalise_gender(raw: str) -> Optional[str]:
    """Duffel accepts only 'm' or 'f' — a constraint of the airline systems
    it fronts, not a design choice of this system (see Chapter 5)."""
    if not raw:
        return None
    text = str(raw).strip().lower()
    if text in ("m", "male", "man"):
        return "m"
    if text in ("f", "female", "woman"):
        return "f"
    return None


def normalise_email(raw: str) -> Optional[str]:
    if not raw:
        return None
    text = str(raw).strip()
    return text if EMAIL_RE.match(text) else None


def normalise_phone(raw: str) -> Optional[str]:
    """Duffel requires E.164 format, e.g. +2305551234."""
    if not raw:
        return None
    text = _re.sub(r"[\s\-()]", "", str(raw).strip())
    return text if E164_RE.match(text) else None


def normalise_name(raw: str) -> Optional[str]:
    if not raw:
        return None
    text = " ".join(str(raw).split())
    if len(text) < 2 or len(text) > 40:
        return None
    if not all(ch.isalpha() or ch in " -'" for ch in text):
        return None
    return text


import uuid


def get_offer(offer_id: str) -> Optional[Dict[str, Any]]:
    """Re-fetch an offer immediately before commitment (FR12).

    Offers go stale. The price shown minutes ago may no longer be the price
    charged, so the offer is re-read and re-confirmed before any order is
    created (failure mode F7).
    """
    try:
        response = requests.get(
            f"{DUFFEL_API_BASE}/air/offers/{offer_id}",
            params={"return_available_services": "false"},
            headers=_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise DuffelError(f"Could not reach Duffel: {exc}") from exc

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise DuffelError(
            f"Duffel returned {response.status_code}: {response.text[:200]}"
        )

    return response.json().get("data")


def new_idempotency_key() -> str:
    return f"bk_{uuid.uuid4().hex}"


def create_order(
    offer_id: str,
    passengers: List[Dict[str, Any]],
    idempotency_key: str,
    order_type: str = "hold",
    total_amount: Optional[str] = None,
    total_currency: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an order. This is the irreversible step.

    order_type is 'hold' (no payment now) or 'instant' (paid from the
    Duffel balance). Card payment via hosted components is out of scope
    for this prototype (see Chapter 5).
    """
    data: Dict[str, Any] = {
        "type": order_type,
        "selected_offers": [offer_id],
        "passengers": passengers,
    }

    if order_type == "instant":
        data["payments"] = [
            {
                "type": "balance",
                "amount": total_amount,
                "currency": total_currency,
            }
        ]

    logger.info(
        "ORDER CREATE | offer=%s | type=%s | key=%s",
        offer_id,
        order_type,
        idempotency_key,
    )

    try:
        response = requests.post(
            f"{DUFFEL_API_BASE}/air/orders",
            headers={
                **_headers(),
                "Content-Type": "application/json",
                "Idempotency-Key": idempotency_key,
            },
            json={"data": data},
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        # The order may or may not have been created. The caller must
        # reconcile rather than retry blindly (F8).
        raise DuffelError(f"Order request did not complete: {exc}") from exc

    if response.status_code not in (200, 201):
        logger.error(
            "ORDER CREATE FAILED | %s | %s", response.status_code, response.text[:500]
        )
        raise DuffelError(
            f"Duffel returned {response.status_code}: {response.text[:300]}"
        )

    return response.json().get("data", {})


def format_confirmation(offer: Dict[str, Any], passenger_line: str) -> str:
    """Render the confirmation summary (FR13).

    Every figure here is read from the provider's response. Nothing about
    price, timing or conditions is generated.
    """
    summary = summarise_offer(offer)
    lines = [
        f"Passenger: {passenger_line}",
        f"Airline: {summary['airline']}",
    ]
    for leg in summary["legs"]:
        lines.append(f"  {format_leg(leg)}")
    lines.append(f"Total: {summary['total_currency']} {summary['total_amount']}")

    conditions = offer.get("conditions") or {}
    def describe(kind: str, label: str) -> str:
        """Render one condition. A null value means the airline did not
        supply the term — which is NOT the same as it being unrestricted.
        Saying so explicitly is the direct mitigation for the class of harm
        in Moffatt v. Air Canada (2024)."""
        cond = conditions.get(kind)
        if cond is None:
            return f"{label}: not stated by the airline — check before you travel"
        if not cond.get("allowed"):
            return f"{label}: not allowed"
        penalty = cond.get("penalty_amount")
        currency = cond.get("penalty_currency") or ""
        if penalty:
            return f"{label}: allowed, fee {currency} {penalty}"
        return f"{label}: allowed, no fee"

    lines.append(describe("change_before_departure", "Changes"))
    lines.append(describe("refund_before_departure", "Refunds"))

    return "\n".join(lines)
