"""Custom CrewAI tools for validating booking request fields."""

import re
from datetime import date, datetime, timedelta
from typing import Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from .flight_tools import AirportLookupTool

MAX_BOOKING_HORIZON_DAYS = 365
MAX_TRIP_DURATION_DAYS = 365
VALID_CABIN_CLASSES = ("economy", "premium_economy", "business", "first")
VALID_TRIP_TYPES = ("one_way", "return", "multi_city")


# --- Date Validation Tool ---

class DateValidationInput(BaseModel):
    date_str: str = Field(description="The date string to validate (ISO format YYYY-MM-DD)")
    field_name: str = Field(description="Which date field: 'departure_date' or 'return_date'")
    departure_date: Optional[str] = Field(
        default=None,
        description="The departure date (ISO), required when validating return_date",
    )


class DateValidationTool(BaseTool):
    name: str = "validate_date"
    description: str = (
        "Validate a date for a flight booking. Checks that the date is real, "
        "not in the past, within the booking horizon (1 year), and for return "
        "dates that it is on or after the departure date."
    )
    args_schema: Type[BaseModel] = DateValidationInput

    def _run(
        self,
        date_str: str,
        field_name: str,
        departure_date: Optional[str] = None,
    ) -> str:
        if not date_str or not date_str.strip():
            return "INVALID: No date provided."

        try:
            parsed = date.fromisoformat(date_str.strip())
        except ValueError:
            return f"INVALID: '{date_str}' is not a valid ISO date (YYYY-MM-DD)."

        today = date.today()

        if parsed < today:
            return f"INVALID: Date {date_str} is in the past (today is {today.isoformat()})."

        if (parsed - today).days > MAX_BOOKING_HORIZON_DAYS:
            return f"INVALID: Date {date_str} is more than {MAX_BOOKING_HORIZON_DAYS} days ahead."

        if field_name == "return_date" and departure_date:
            try:
                dep = date.fromisoformat(departure_date)
                if parsed < dep:
                    return f"INVALID: Return date {date_str} is before departure date {departure_date}."
                if (parsed - dep).days > MAX_TRIP_DURATION_DAYS:
                    return f"INVALID: Trip duration exceeds {MAX_TRIP_DURATION_DAYS} days."
            except ValueError:
                return f"INVALID: Departure date '{departure_date}' is not valid ISO format."

        return f"VALID: {date_str} is acceptable for {field_name}."


# --- Passenger Count Validation Tool ---

class PassengerCountInput(BaseModel):
    count: str = Field(description="The passenger count value to validate")


class PassengerCountValidationTool(BaseTool):
    name: str = "validate_passenger_count"
    description: str = (
        "Validate passenger count for a flight booking. "
        "Must be an integer between 1 and 9."
    )
    args_schema: Type[BaseModel] = PassengerCountInput

    def _run(self, count: str) -> str:
        if not count:
            return "INVALID: No passenger count provided."

        try:
            n = int(float(count))
        except (TypeError, ValueError):
            return f"INVALID: '{count}' is not a valid number."

        if n < 1:
            return "INVALID: Passenger count must be at least 1."
        if n > 9:
            return "INVALID: Maximum 9 passengers per booking."

        return f"VALID: {n} passenger(s)."


# --- Cabin Class Validation Tool ---

class CabinClassInput(BaseModel):
    cabin_class: str = Field(description="The cabin class to validate")


class CabinClassValidationTool(BaseTool):
    name: str = "validate_cabin_class"
    description: str = (
        "Validate cabin class. Must be one of: economy, premium_economy, business, first."
    )
    args_schema: Type[BaseModel] = CabinClassInput

    def _run(self, cabin_class: str) -> str:
        if not cabin_class:
            return "INVALID: No cabin class provided."

        normalized = cabin_class.strip().lower().replace(" ", "_").replace("-", "_")

        # Common mappings
        mappings = {
            "economy_class": "economy",
            "coach": "economy",
            "premium": "premium_economy",
            "premium_economy_class": "premium_economy",
            "business_class": "business",
            "first_class": "first",
        }
        normalized = mappings.get(normalized, normalized)

        if normalized not in VALID_CABIN_CLASSES:
            return (
                f"INVALID: '{cabin_class}' is not a valid cabin class. "
                f"Must be one of: {', '.join(VALID_CABIN_CLASSES)}."
            )

        return f"VALID: cabin_class={normalized}"


# --- Trip Type Validation Tool ---

class TripTypeInput(BaseModel):
    trip_type: str = Field(description="The trip type to validate")
    return_date: Optional[str] = Field(
        default=None,
        description="Return date if provided, to check consistency",
    )


class TripTypeValidationTool(BaseTool):
    name: str = "validate_trip_type"
    description: str = (
        "Validate trip type. Must be one of: one_way, return, multi_city. "
        "Also checks consistency: return trips must have a return date."
    )
    args_schema: Type[BaseModel] = TripTypeInput

    def _run(self, trip_type: str, return_date: Optional[str] = None) -> str:
        if not trip_type:
            return "INVALID: No trip type provided."

        normalized = trip_type.strip().lower().replace(" ", "_").replace("-", "_")

        mappings = {
            "round_trip": "return",
            "roundtrip": "return",
            "oneway": "one_way",
            "one_way_trip": "one_way",
        }
        normalized = mappings.get(normalized, normalized)

        if normalized not in VALID_TRIP_TYPES:
            return (
                f"INVALID: '{trip_type}' is not a valid trip type. "
                f"Must be one of: {', '.join(VALID_TRIP_TYPES)}."
            )

        if normalized == "return" and not return_date:
            return "WARNING: Trip type is 'return' but no return date provided. A return date is required."

        if normalized == "one_way" and return_date:
            return "WARNING: Trip type is 'one_way' but a return date was provided. The return date will be ignored."

        return f"VALID: trip_type={normalized}"


# --- Route Validation Tool ---

class RouteValidationInput(BaseModel):
    origin_code: str = Field(description="Origin IATA airport code (3 letters)")
    destination_code: str = Field(description="Destination IATA airport code (3 letters)")


class RouteValidationTool(BaseTool):
    name: str = "validate_route"
    description: str = (
        "Validate that origin and destination are different valid IATA codes. "
        "Checks format (3 uppercase letters) and that they are not the same."
    )
    args_schema: Type[BaseModel] = RouteValidationInput

    def _run(self, origin_code: str, destination_code: str) -> str:
        iata_pattern = re.compile(r"^[A-Z]{3}$")

        origin = (origin_code or "").strip().upper()
        destination = (destination_code or "").strip().upper()

        errors = []

        if not origin:
            errors.append("Origin airport code is missing.")
        elif not iata_pattern.match(origin):
            errors.append(f"Origin '{origin_code}' is not a valid 3-letter IATA code.")

        if not destination:
            errors.append("Destination airport code is missing.")
        elif not iata_pattern.match(destination):
            errors.append(f"Destination '{destination_code}' is not a valid 3-letter IATA code.")

        if origin and destination and origin == destination:
            errors.append(f"Origin and destination are the same ({origin}).")

        if errors:
            return "INVALID: " + " ".join(errors)

        return f"VALID: Route {origin} → {destination} is well-formed."


# --- Build Duffel Request Tool ---

class BuildRequestInput(BaseModel):
    origin_code: str = Field(description="Origin IATA code")
    destination_code: str = Field(description="Destination IATA code")
    departure_date: str = Field(description="Departure date YYYY-MM-DD")
    return_date: Optional[str] = Field(default=None, description="Return date YYYY-MM-DD")
    passengers: int = Field(description="Number of adult passengers")
    cabin_class: str = Field(description="Cabin class")


class BuildDuffelRequestTool(BaseTool):
    name: str = "build_duffel_request"
    description: str = (
        "Build a valid Duffel API offer request payload from validated fields. "
        "Only call this after all fields have passed validation."
    )
    args_schema: Type[BaseModel] = BuildRequestInput

    def _run(
        self,
        origin_code: str,
        destination_code: str,
        departure_date: str,
        return_date: Optional[str] = None,
        passengers: int = 1,
        cabin_class: str = "economy",
    ) -> str:
        slices = [
            {
                "origin": origin_code.upper(),
                "destination": destination_code.upper(),
                "departure_date": departure_date,
            }
        ]
        if return_date:
            slices.append(
                {
                    "origin": destination_code.upper(),
                    "destination": origin_code.upper(),
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

        import json
        return f"DUFFEL_REQUEST_READY: {json.dumps(payload, indent=2)}"
