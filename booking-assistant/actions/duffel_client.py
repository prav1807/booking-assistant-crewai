import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DUFFEL_API_BASE = "https://api.duffel.com"
DUFFEL_VERSION = "v2"
REQUEST_TIMEOUT_SECONDS = 10

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
            a for a in top.get("airports", [])
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


def match_choice(choice: str, options: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
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
        logger.error("OFFER REQUEST FAILED | %s | %s",
                     response.status_code, response.text[:500])
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
    }