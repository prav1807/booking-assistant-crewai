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
        # Use the policy_question slot (filled by the flow), fallback to latest message
        question = tracker.get_slot("policy_question") or tracker.latest_message.get("text", "")

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


class ActionVerifyBooking(Action):
    """BookingReviewCrew (Role 5) — verify order payload before submission."""

    def name(self) -> Text:
        return "action_verify_booking"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        payload = {
            "order_payload": tracker.get_slot("duffel_request_payload") or {},
            "passenger_details": {
                "given_name": tracker.get_slot("given_name"),
                "family_name": tracker.get_slot("family_name"),
                "born_on": tracker.get_slot("born_on"),
                "title": tracker.get_slot("passenger_title"),
                "gender": tracker.get_slot("passenger_gender"),
                "email": tracker.get_slot("passenger_email"),
                "phone": tracker.get_slot("passenger_phone"),
            },
            "selected_offer": {
                "id": tracker.get_slot("selected_offer_id"),
            },
            "conversation_slots": {
                "origin_code": tracker.get_slot("origin_code"),
                "destination_code": tracker.get_slot("destination_code"),
                "selected_offer_id": tracker.get_slot("selected_offer_id"),
                "passenger_count": tracker.get_slot("passenger_count"),
                "given_name": tracker.get_slot("given_name"),
                "family_name": tracker.get_slot("family_name"),
                "born_on": tracker.get_slot("born_on"),
                "passenger_title": tracker.get_slot("passenger_title"),
                "passenger_gender": tracker.get_slot("passenger_gender"),
                "passenger_email": tracker.get_slot("passenger_email"),
                "passenger_phone": tracker.get_slot("passenger_phone"),
            },
        }

        logger.info("ACTION_VERIFY_BOOKING | offer_id=%s", tracker.get_slot("selected_offer_id"))
        result = _post_crew("/verify", payload)

        if "error" in result:
            return [SlotSet("booking_verified", False),
                    SlotSet("verification_reason", "Verification service unavailable.")]

        return [
            SlotSet("booking_verified", result.get("verified", False)),
            SlotSet("verification_reason", result.get("reason", "")),
        ]


class ActionGuardrailCheck(Action):
    """GuardrailCrew (Role 6) — screen outbound messages for safety."""

    def name(self) -> Text:
        return "action_guardrail_check"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        draft = tracker.get_slot("pending_bot_message") or ""

        if not draft:
            return [SlotSet("guardrail_passed", True)]

        logger.info("ACTION_GUARDRAIL_CHECK | message_len=%d", len(draft))
        result = _post_crew("/guardrail", {"draft_message": draft})

        if "error" in result:
            return [SlotSet("guardrail_passed", True)]

        return [
            SlotSet("guardrail_passed", result.get("passed", True)),
            SlotSet("guardrail_reason", result.get("reason", "")),
        ]


class ActionRetrieveBooking(Action):
    """Retrieve a booking by reference + email verification."""

    def name(self) -> Text:
        return "action_retrieve_booking"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        reference = tracker.get_slot("booking_lookup_reference")
        email = tracker.get_slot("booking_lookup_email")

        logger.info("ACTION_RETRIEVE_BOOKING | ref=%s | email=%s", reference, email)

        if not reference or not email:
            return [SlotSet("booking_retrieval_status", "not_found")]

        # Look up booking in Supabase
        try:
            from .supabase_client import get_booking_by_reference
            booking = get_booking_by_reference(reference)
        except Exception as exc:
            logger.error("Booking lookup failed: %s", exc)
            return [SlotSet("booking_retrieval_status", "not_found")]

        if not booking:
            return [SlotSet("booking_retrieval_status", "not_found")]

        # Verify email matches — prevents unauthorized access
        booking_email = booking.get("passenger_email", "").strip().lower()
        lookup_email = email.strip().lower()

        if booking_email != lookup_email:
            logger.warning("BOOKING AUTH FAILED | ref=%s | expected=%s | got=%s",
                          reference, booking_email, lookup_email)
            return [SlotSet("booking_retrieval_status", "unauthorized")]

        # Build the details text
        details = (
            f"Reference: {booking.get('booking_reference', reference)}\n"
            f"Route: {booking.get('origin_code', '?')} -> {booking.get('destination_code', '?')}\n"
            f"Departure: {booking.get('departure_date', '?')}\n"
            f"Passenger: {booking.get('passenger_title', '')} "
            f"{booking.get('given_name', '')} {booking.get('family_name', '')}\n"
            f"Status: {booking.get('status', 'confirmed')}"
        )

        return [
            SlotSet("booking_retrieval_status", "found"),
            SlotSet("booking_details_text", details),
            SlotSet("pending_bot_message", details),
        ]


class ActionGuardrailBookingResponse(Action):
    """Screen booking details through guardrail before showing to user."""

    def name(self) -> Text:
        return "action_guardrail_booking_response"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[EventType]:
        details = tracker.get_slot("booking_details_text") or ""

        if not details:
            return [SlotSet("guardrail_passed", True)]

        logger.info("ACTION_GUARDRAIL_BOOKING | screening booking details")
        result = _post_crew("/guardrail", {
            "draft_message": details,
            "context": {"type": "booking_retrieval"},
        })

        if "error" in result:
            # Fail-closed for booking retrieval (unlike general guardrail)
            return [SlotSet("guardrail_passed", False),
                    SlotSet("guardrail_reason", "Security check unavailable")]

        return [
            SlotSet("guardrail_passed", result.get("passed", False)),
            SlotSet("guardrail_reason", result.get("reason", "")),
        ]
