"""Custom CrewAI tools that wrap the Duffel flight search API."""

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import requests
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DUFFEL_API_BASE = os.getenv("DUFFEL_BASE_URL", "https://api.duffel.com")
DUFFEL_VERSION = "v2"
SEARCH_TIMEOUT = 60


def _headers() -> Dict[str, str]:
    token = os.getenv("DUFFEL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("DUFFEL_ACCESS_TOKEN not set in .env")
    return {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


# --- Flight Search Tool ---

class FlightSearchInput(BaseModel):
    origin: str = Field(description="IATA airport code for departure (e.g. 'MRU', 'LHR')")
    destination: str = Field(description="IATA airport code for arrival (e.g. 'CDG', 'DXB')")
    departure_date: str = Field(description="Departure date in ISO format YYYY-MM-DD")
    return_date: Optional[str] = Field(
        default=None,
        description="Return date in ISO format YYYY-MM-DD. Leave empty for one-way.",
    )
    passengers: int = Field(default=1, description="Number of adult passengers")
    cabin_class: str = Field(
        default="economy",
        description="Cabin class: economy, premium_economy, business, or first",
    )


class FlightSearchTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search for available flights between two airports on given dates. "
        "Returns a list of flight offers with prices, airlines, durations, and stops. "
        "Use IATA airport codes (3 letters) for origin and destination."
    )
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        passengers: int = 1,
        cabin_class: str = "economy",
    ) -> str:
        slices = [
            {
                "origin": origin.upper(),
                "destination": destination.upper(),
                "departure_date": departure_date,
            }
        ]
        if return_date:
            slices.append(
                {
                    "origin": destination.upper(),
                    "destination": origin.upper(),
                    "departure_date": return_date,
                }
            )

        payload = {
            "data": {
                "slices": slices,
                "passengers": [{"type": "adult"} for _ in range(passengers)],
                "cabin_class": cabin_class,
            }
        }

        try:
            response = requests.post(
                f"{DUFFEL_API_BASE}/air/offer_requests",
                params={"return_offers": "true"},
                headers=_headers(),
                json=payload,
                timeout=SEARCH_TIMEOUT,
            )
        except requests.RequestException as exc:
            return f"Error: Could not reach Duffel API — {exc}"

        if response.status_code not in (200, 201):
            return f"Error: Duffel returned {response.status_code} — {response.text[:200]}"

        offers = response.json().get("data", {}).get("offers", [])
        if not offers:
            return "No flights found for this route and date combination."

        offers.sort(key=lambda o: float(o.get("total_amount") or 1e12))
        return self._format_offers(offers[:10])

    def _format_offers(self, offers: List[Dict[str, Any]]) -> str:
        lines = [f"Found {len(offers)} flight option(s):\n"]

        for i, offer in enumerate(offers, 1):
            airline = (offer.get("owner") or {}).get("name", "Unknown")
            amount = offer.get("total_amount", "?")
            currency = offer.get("total_currency", "")
            expires = offer.get("expires_at", "")

            legs_info = []
            for sl in offer.get("slices", []):
                segments = sl.get("segments", [])
                if not segments:
                    continue
                first, last = segments[0], segments[-1]
                orig = (first.get("origin") or {}).get("iata_code", "?")
                dest = (last.get("destination") or {}).get("iata_code", "?")
                dep = first.get("departing_at", "")[:16]
                arr = last.get("arriving_at", "")[:16]
                stops = max(len(segments) - 1, 0)
                stop_text = "direct" if stops == 0 else f"{stops} stop(s)"
                duration = sl.get("duration", "")
                legs_info.append(
                    f"    {orig} → {dest} | {dep} – {arr} | {duration} | {stop_text}"
                )

            lines.append(
                f"{i}. {airline} — {currency} {amount}\n" + "\n".join(legs_info)
            )

        return "\n\n".join(lines)


# --- Airport Lookup Tool ---

class AirportLookupInput(BaseModel):
    query: str = Field(description="City name or airport name to look up IATA codes for")


class AirportLookupTool(BaseTool):
    name: str = "lookup_airport"
    description: str = (
        "Look up airport IATA codes for a city or airport name. "
        "Use this when you only have a city name and need the 3-letter IATA code "
        "before searching for flights."
    )
    args_schema: Type[BaseModel] = AirportLookupInput

    def _run(self, query: str) -> str:
        if not query or not query.strip():
            return "No airports found for ''. Please provide a city or airport name."

        try:
            response = requests.get(
                f"{DUFFEL_API_BASE}/places/suggestions",
                params={"query": query.strip()},
                headers=_headers(),
                timeout=10,
            )
        except requests.RequestException as exc:
            return f"Error looking up airport: {exc}"

        if response.status_code != 200:
            return f"No airports found for '{query}'."

        places = response.json().get("data", [])
        if not places:
            return f"No airports found for '{query}'."

        results = []
        for place in places[:5]:
            ptype = place.get("type", "")
            if ptype == "airport":
                code = place.get("iata_code", "")
                name = place.get("name", "")
                city = place.get("city_name", "")
                results.append(f"  {code} — {name} ({city})")
            elif ptype == "city":
                city_name = place.get("name", "")
                airports = place.get("airports", [])
                for a in airports[:5]:
                    code = a.get("iata_code", "")
                    name = a.get("name", "")
                    if code:
                        results.append(f"  {code} — {name} ({city_name})")

        if not results:
            return f"No airports found for '{query}'."

        return f"Airports matching '{query}':\n" + "\n".join(results)
