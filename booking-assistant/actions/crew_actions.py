"""Rasa custom actions that integrate with the CrewAI crew service.

These actions are THE SEAM between Rasa CALM and the CrewAI multi-agent layer.
Each action POSTs to the FastAPI crew service and returns structured slot updates.
The crew service returns verdicts — never user-facing prose.
"""

import logging
import os
from typing import Any, Dict, List, Text

import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

CREW_SERVICE_URL = os.getenv("CREW_SERVICE_URL", "http://localhost:8000")
CREW_SERVICE_TIMEOUT = int(os.getenv("CREW_SERVICE_TIMEOUT", "120"))


def _post_crew(endpoint: str, payload: dict) -> dict:
    """POST to the crew service with timeout and error handling."""
    url = f"{CREW_SERVICE_URL}{endpoint}"
    try:
        response = requests.post(url, json=payload, timeout=CREW_SERVICE_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        logger.error("Crew service timeout: %s", url)
        return {"error": "timeout"}
    except requests.ConnectionError:
        logger.error("Crew service unavailable: %s", url)
        return {"error": "connection_error"}
    except requests.HTTPError as exc:
        logger.error("Crew service error: %s | %s", url, exc)
        return {"error": str(exc)}


class ActionValidateBooking(Action):
    """Invoke the ValidationCrew to validate all collected booking slots.

    Corresponds to the report's Role 1: validate, disambiguate, build request.
    Returns structured verdict → flow branches on validation_passed.
    """

    def name(self) -> Text:
        return "action_validate_booking"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        slots = {
            "origin_code": tracker.get_slot("origin_code"),
            "destination_code": tracker.get_slot("destination_code"),
            "departure_date": tracker.get_slot("departure_date"),
            "return_date": tracker.get_slot("return_date"),
            "passenger_count": tracker.get_slot("passenger_count"),
            "cabin_class": tracker.get_slot("cabin_class"),
            "trip_type": tracker.get_slot("trip_type"),
        }

        logger.info("ACTION_VALIDATE_BOOKING | slots=%s", slots)

        result = _post_crew("/validate", slots)

        if "error" in result:
            logger.error("ValidationCrew unavailable: %s", result["error"])
            return [SlotSet("validation_passed", False),
                    SlotSet("validation_reason", "The validation service is temporarily unavailable. Please try again.")]

        events = [
            SlotSet("validation_passed", result.get("valid", False)),
        ]

        if not result.get("valid"):
            errors = result.get("errors", [])
            reason = "; ".join(errors) if errors else "Validation failed."
            events.append(SlotSet("validation_reason", reason))
        else:
            events.append(SlotSet("validation_reason", None))
            duffel_req = result.get("duffel_request")
            if duffel_req:
                events.append(SlotSet("duffel_request_payload", duffel_req))

        # Apply corrected slots
        for field, value in result.get("corrected_slots", {}).items():
            events.append(SlotSet(field, value))

        warnings = result.get("warnings", [])
        if warnings:
            events.append(SlotSet("validation_warnings", warnings))

        logger.info("ACTION_VALIDATE_BOOKING | valid=%s | errors=%s",
                    result.get("valid"), result.get("errors"))

        return events


class ActionCrewSearch(Action):
    """Invoke the FlightSearchCrew to search and rank flights.

    Corresponds to the report's Roles 2+3: search strategy and ranking.
    """

    def name(self) -> Text:
        return "action_crew_search"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        origin = tracker.get_slot("origin_code") or ""
        destination = tracker.get_slot("destination_code") or ""
        departure = tracker.get_slot("departure_date") or ""
        return_date = tracker.get_slot("return_date") or ""
        passengers = tracker.get_slot("passenger_count") or 1
        cabin = tracker.get_slot("cabin_class") or "economy"
        trip_type = tracker.get_slot("trip_type") or "one_way"

        user_request = (
            f"{'Return' if trip_type == 'return' else 'One-way'} flights from "
            f"{origin} to {destination}, departing {departure}"
            f"{f', returning {return_date}' if return_date else ''}, "
            f"{cabin} class, {int(float(passengers))} passenger(s)"
        )

        logger.info("ACTION_CREW_SEARCH | request=%s", user_request)

        result = _post_crew("/search", {"user_request": user_request})

        if "error" in result:
            logger.error("FlightSearchCrew unavailable: %s", result["error"])
            return [SlotSet("crew_search_status", "error")]

        if result.get("success"):
            return [
                SlotSet("crew_search_status", "ok"),
                SlotSet("crew_search_result", result.get("result", "")),
            ]
        else:
            return [
                SlotSet("crew_search_status", "error"),
                SlotSet("crew_search_result", result.get("error", "")),
            ]


class ActionAnswerPolicy(Action):
    """Invoke the PolicyCrew to answer a policy or visa question.

    Corresponds to the report's Role 4: grounded policy answering.
    Invoked when CALM emits a KnowledgeAnswer command.
    """

    def name(self) -> Text:
        return "action_answer_policy"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        # Get the latest user message as the policy question
        question = tracker.latest_message.get("text", "")

        logger.info("ACTION_ANSWER_POLICY | question=%s", question[:100])

        payload = {
            "question": question,
            "passport_country": tracker.get_slot("passport_country"),
            "destination_country": tracker.get_slot("destination_country"),
        }

        result = _post_crew("/policy", payload)

        if "error" in result:
            logger.error("PolicyCrew unavailable: %s", result["error"])
            dispatcher.utter_message(
                text="I'm sorry, I'm unable to look that up right now. Please try again shortly."
            )
            return []

        if result.get("answered"):
            dispatcher.utter_message(text=result.get("answer", ""))
        else:
            dispatcher.utter_message(
                text="I don't have information about that in my knowledge base. "
                     "Would you like me to connect you with a human agent?"
            )

        return [
            SlotSet("policy_answer", result.get("answer")),
            SlotSet("policy_sources", result.get("sources", [])),
        ]
