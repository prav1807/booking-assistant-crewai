"""ValidationCrew — validates booking request fields and builds the Duffel request.

This crew contains two agents:
  1. RequestValidatorAgent — validates all collected slots (origin, destination,
     dates, passengers, cabin class, trip type) and flags errors.
  2. RequestBuilderAgent — assembles a valid Duffel API request payload from
     the validated fields.

Returns a structured verdict to the calling system (Rasa flow), never user-facing prose.
"""

import json
from typing import Any, Dict, Optional

from crewai import Agent, Crew, Process, Task

from .llm_config import build_llm
from .tools.flight_tools import AirportLookupTool
from .tools.validation_tools import (
    BuildDuffelRequestTool,
    CabinClassValidationTool,
    DateValidationTool,
    PassengerCountValidationTool,
    RouteValidationTool,
    TripTypeValidationTool,
)


def create_validator_agent() -> Agent:
    """Agent that validates all booking request fields."""
    return Agent(
        role="Request Validator",
        goal=(
            "Validate every field of a flight booking request. Check that all "
            "required fields are present and correctly formatted. Report any "
            "errors with clear reasons."
        ),
        backstory=(
            "You are a meticulous data validation specialist for an airline "
            "booking system. You check each field against strict rules: "
            "IATA codes must be 3 uppercase letters, dates must be valid and "
            "in the future, passenger count must be 1-9, cabin class must be "
            "one of the allowed values. You never let invalid data through."
        ),
        tools=[
            AirportLookupTool(),
            DateValidationTool(),
            PassengerCountValidationTool(),
            CabinClassValidationTool(),
            TripTypeValidationTool(),
            RouteValidationTool(),
        ],
        llm=build_llm(),
        verbose=True,
        max_iter=5,
    )


def create_builder_agent() -> Agent:
    """Agent that builds the Duffel API request from validated fields."""
    return Agent(
        role="Request Builder",
        goal=(
            "Assemble a correctly structured Duffel API offer request payload "
            "from the validated booking fields. Only build the request if all "
            "fields have been validated successfully."
        ),
        backstory=(
            "You are a systems integration specialist. Your job is to take "
            "validated booking parameters and construct the exact JSON payload "
            "that the Duffel API expects. You never build a request from "
            "invalid data."
        ),
        tools=[BuildDuffelRequestTool()],
        llm=build_llm(),
        verbose=True,
        max_iter=3,
    )


def create_validation_task(agent: Agent, slots: Dict[str, Any]) -> Task:
    """Create the validation task from collected slots."""
    slot_summary = "\n".join(f"  - {k}: {v}" for k, v in slots.items() if v is not None)

    return Task(
        description=(
            f"Validate the following booking request fields:\n{slot_summary}\n\n"
            "Steps:\n"
            "1. Validate the route: check origin and destination are valid, "
            "   different IATA codes. If they look like city names, look up the codes.\n"
            "2. Validate departure_date: must be a valid future date within 1 year.\n"
            "3. If trip_type is 'return', validate return_date: must be on or after departure.\n"
            "4. Validate passenger_count: integer 1-9.\n"
            "5. Validate cabin_class: must be economy, premium_economy, business, or first.\n"
            "6. Validate trip_type: must be one_way, return, or multi_city.\n"
            "7. Check consistency: return trip needs return_date, one_way should not have one.\n\n"
            "Output a JSON verdict with this structure:\n"
            '{"valid": true/false, "errors": [...], "corrected_slots": {...}, "warnings": [...]}'
        ),
        expected_output=(
            "A JSON object with keys: valid (bool), errors (list of strings), "
            "corrected_slots (dict of field->corrected_value), warnings (list of strings)."
        ),
        agent=agent,
    )


def create_build_task(agent: Agent, slots: Dict[str, Any]) -> Task:
    """Create the request building task."""
    return Task(
        description=(
            "Using the validated booking fields, build the Duffel API request payload.\n"
            f"Fields: origin_code={slots.get('origin_code')}, "
            f"destination_code={slots.get('destination_code')}, "
            f"departure_date={slots.get('departure_date')}, "
            f"return_date={slots.get('return_date')}, "
            f"passengers={slots.get('passenger_count', 1)}, "
            f"cabin_class={slots.get('cabin_class', 'economy')}\n\n"
            "Use the build_duffel_request tool to construct the payload."
        ),
        expected_output="The complete Duffel API request payload as JSON.",
        agent=agent,
    )


def run_validation_crew(slots: Dict[str, Any]) -> Dict[str, Any]:
    """Run the ValidationCrew on collected booking slots.

    Args:
        slots: Dictionary of booking fields, e.g.:
            {
                "origin": "London" or "LHR",
                "origin_code": "LHR",
                "destination": "New York" or "JFK",
                "destination_code": "JFK",
                "departure_date": "2026-10-01",
                "return_date": "2026-10-15",  # optional
                "passenger_count": 2,
                "cabin_class": "business",
                "trip_type": "return",
            }

    Returns:
        A structured verdict dict:
            {
                "valid": bool,
                "errors": [...],
                "warnings": [...],
                "corrected_slots": {...},
                "duffel_request": {...} or None,
            }
    """
    # Step 0: Run deterministic tool-level validation (fast, no LLM)
    errors, warnings = _run_deterministic_checks(slots)

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "corrected_slots": {},
            "duffel_request": None,
            "raw_output": "",
        }

    # Step 1: Run the LLM-powered validator for deeper checks (ambiguity, etc.)
    validator = create_validator_agent()
    validation_task = create_validation_task(validator, slots)

    validation_crew = Crew(
        agents=[validator],
        tasks=[validation_task],
        process=Process.sequential,
        verbose=True,
    )

    val_result = validation_crew.kickoff()
    val_output_parts = [str(val_result)]
    if hasattr(val_result, "tasks_output"):
        for task_output in val_result.tasks_output:
            val_output_parts.append(str(task_output))
    val_output = "\n".join(val_output_parts)

    # Step 2: Build the Duffel request
    builder = create_builder_agent()
    build_task = create_build_task(builder, slots)

    build_crew = Crew(
        agents=[builder],
        tasks=[build_task],
        process=Process.sequential,
        verbose=True,
    )

    build_result = build_crew.kickoff()
    build_output_parts = [str(build_result)]
    if hasattr(build_result, "tasks_output"):
        for task_output in build_result.tasks_output:
            build_output_parts.append(str(task_output))
    build_output = "\n".join(build_output_parts)

    full_output = val_output + "\n" + build_output
    verdict = _parse_verdict(full_output, slots)
    verdict["warnings"] = warnings
    return verdict


def _run_deterministic_checks(slots: Dict[str, Any]) -> tuple:
    """Run fast, deterministic validation without LLM. Returns (errors, warnings)."""
    from .tools.validation_tools import (
        CabinClassValidationTool,
        DateValidationTool,
        PassengerCountValidationTool,
        RouteValidationTool,
        TripTypeValidationTool,
    )

    errors = []
    warnings = []

    # Route validation
    origin = slots.get("origin_code", "")
    destination = slots.get("destination_code", "")
    if origin and destination:
        rv = RouteValidationTool()
        result = rv._run(origin, destination)
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())

    # Departure date
    dep_date = slots.get("departure_date", "")
    if dep_date:
        dt = DateValidationTool()
        result = dt._run(str(dep_date), "departure_date")
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())

    # Return date
    ret_date = slots.get("return_date")
    if ret_date:
        dt = DateValidationTool()
        result = dt._run(str(ret_date), "return_date", str(dep_date) if dep_date else None)
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())

    # Passenger count
    pax = slots.get("passenger_count")
    if pax is not None:
        pc = PassengerCountValidationTool()
        result = pc._run(str(pax))
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())

    # Cabin class
    cabin = slots.get("cabin_class", "")
    if cabin:
        cc = CabinClassValidationTool()
        result = cc._run(str(cabin))
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())

    # Trip type
    trip = slots.get("trip_type", "")
    if trip:
        tt = TripTypeValidationTool()
        result = tt._run(str(trip), str(ret_date) if ret_date else None)
        if "INVALID:" in result:
            errors.append(result.split("INVALID:")[-1].strip())
        elif "WARNING:" in result:
            warnings.append(result.split("WARNING:")[-1].strip())

    return errors, warnings

    # Try to parse structured output
    verdict = _parse_verdict(raw_output, slots)
    return verdict


def _parse_verdict(raw_output: str, slots: Dict[str, Any]) -> Dict[str, Any]:
    """Parse the crew's output into a structured verdict."""
    verdict = {
        "valid": False,
        "errors": [],
        "warnings": [],
        "corrected_slots": {},
        "duffel_request": None,
        "raw_output": raw_output,
    }

    # Check for INVALID markers
    if "INVALID:" in raw_output:
        for line in raw_output.split("\n"):
            if "INVALID:" in line:
                error_msg = line.split("INVALID:")[-1].strip()
                if error_msg and error_msg not in verdict["errors"]:
                    verdict["errors"].append(error_msg)

    # Check for WARNING markers
    if "WARNING:" in raw_output:
        for line in raw_output.split("\n"):
            if "WARNING:" in line:
                warn_msg = line.split("WARNING:")[-1].strip()
                if warn_msg and warn_msg not in verdict["warnings"]:
                    verdict["warnings"].append(warn_msg)

    # Check if the Duffel request was built (indicates validation passed)
    if "DUFFEL_REQUEST_READY" in raw_output:
        verdict["valid"] = True
        try:
            start = raw_output.index("DUFFEL_REQUEST_READY:") + len("DUFFEL_REQUEST_READY:")
            json_str = raw_output[start:].strip()
            brace_start = json_str.index("{")
            depth = 0
            for i, ch in enumerate(json_str[brace_start:], brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = json_str[brace_start:i + 1]
                        break
            verdict["duffel_request"] = json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass
    elif not verdict["errors"]:
        # If no INVALID found and no DUFFEL_REQUEST either, check for "slices"
        # which indicates the builder produced a payload in its own format
        if '"slices"' in raw_output and '"passengers"' in raw_output:
            verdict["valid"] = True
            try:
                # Find JSON with slices
                idx = raw_output.index('"slices"')
                # Walk back to find opening brace
                search_back = raw_output[:idx]
                brace_pos = search_back.rfind("{")
                if brace_pos >= 0:
                    remaining = raw_output[brace_pos:]
                    depth = 0
                    for i, ch in enumerate(remaining):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                candidate = remaining[:i + 1]
                                verdict["duffel_request"] = json.loads(candidate)
                                break
            except (ValueError, json.JSONDecodeError):
                pass

    # If there are errors, force valid=False regardless of other signals
    if verdict["errors"]:
        verdict["valid"] = False

    return verdict
