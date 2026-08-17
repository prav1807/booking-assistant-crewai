"""FastAPI Crew Service — exposes CrewAI crews as HTTP endpoints.

Per the report (Section 4.4), this is THE SEAM between Rasa and CrewAI:
- Rasa custom actions POST to these endpoints
- Crews return structured verdicts, never user-facing prose
- All user-facing text is rendered by CALM flow responses

Endpoints:
    POST /validate  — ValidationCrew (Role 1)
    POST /search    — FlightSearchCrew (Roles 2+3)
    POST /policy    — PolicyCrew (Role 4)
    GET  /health    — Health check
"""

import os
import sys
import logging
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# Ensure crewai_agents is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Crew Service",
    description="CrewAI multi-agent service for the flight booking assistant",
    version="1.0.0",
)


# --- Request/Response Models ---

class ValidationRequest(BaseModel):
    origin_code: Optional[str] = None
    destination_code: Optional[str] = None
    departure_date: Optional[str] = None
    return_date: Optional[str] = None
    passenger_count: Optional[int] = None
    cabin_class: Optional[str] = None
    trip_type: Optional[str] = None


class ValidationResponse(BaseModel):
    valid: bool
    errors: list = []
    warnings: list = []
    corrected_slots: dict = {}
    duffel_request: Optional[dict] = None


class SearchRequest(BaseModel):
    user_request: str


class SearchResponse(BaseModel):
    success: bool
    result: str
    error: Optional[str] = None


class PolicyRequest(BaseModel):
    question: str
    passport_country: Optional[str] = None
    destination_country: Optional[str] = None


class PolicyResponse(BaseModel):
    answered: bool
    answer: str
    sources: list = []
    question_type: str = "policy"


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "crew-service"}


@app.post("/validate", response_model=ValidationResponse)
def validate_booking(request: ValidationRequest):
    """Invoke the ValidationCrew to validate booking slots and build the Duffel request."""
    logger.info("POST /validate | slots=%s", request.model_dump())

    from crewai_agents.validation_crew import run_validation_crew

    slots = request.model_dump(exclude_none=False)
    try:
        verdict = run_validation_crew(slots)
    except Exception as exc:
        logger.error("ValidationCrew failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Validation crew error: {exc}")

    return ValidationResponse(
        valid=verdict.get("valid", False),
        errors=verdict.get("errors", []),
        warnings=verdict.get("warnings", []),
        corrected_slots=verdict.get("corrected_slots", {}),
        duffel_request=verdict.get("duffel_request"),
    )


@app.post("/search", response_model=SearchResponse)
def search_flights(request: SearchRequest):
    """Invoke the FlightSearchCrew to search and rank flights."""
    logger.info("POST /search | request=%s", request.user_request[:100])

    from crewai_agents.flight_search_crew import run_flight_search

    try:
        result = run_flight_search(request.user_request)
        return SearchResponse(success=True, result=result)
    except Exception as exc:
        logger.error("FlightSearchCrew failed: %s", exc)
        return SearchResponse(success=False, result="", error=str(exc))


@app.post("/policy", response_model=PolicyResponse)
def answer_policy(request: PolicyRequest):
    """Invoke the PolicyCrew to answer a policy or visa eligibility question."""
    logger.info("POST /policy | question=%s", request.question[:100])

    from crewai_agents.policy_crew import run_policy_crew

    try:
        result = run_policy_crew(
            question=request.question,
            passport_country=request.passport_country,
            destination_country=request.destination_country,
        )
    except Exception as exc:
        logger.error("PolicyCrew failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Policy crew error: {exc}")

    return PolicyResponse(
        answered=result.get("answered", False),
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        question_type=result.get("question_type", "policy"),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
