import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Text

import dateparser
from matplotlib.pylab import choice
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher

from .duffel_client import (
    DuffelError,
    MAX_OFFERS_SHOWN,
    build_slices,
    create_offer_request,
    format_offer,
    format_offer_list,
    format_options,
    match_choice,
    minutes_until_expiry,
    resolve_place,
    summarise_offer,
    normalise_email,
    normalise_gender,
    normalise_name,
    normalise_phone,
    normalise_title,
)

logger = logging.getLogger(__name__)

MAX_OFFERS_STORED = 20
MAX_BOOKING_HORIZON_DAYS = 365  # how far ahead a departure may be booked
MAX_TRIP_DURATION_DAYS = 365  # how long a trip may last, from departure


def normalise_date(
    raw: Any,
    not_before: Optional[date] = None,
    max_days_ahead: int = MAX_BOOKING_HORIZON_DAYS,
) -> Optional[str]:
    """Return an ISO date string, or None if the input is unusable.

    None is the signal for 'invalid' — the flow branches on the slot being
    empty, so no separate validity flag is needed.
    """
    if raw in (None, ""):
        return None

    today = date.today()
    floor = not_before or today

    text = str(raw).strip()

    # The command generator often normalises dates to ISO before this action
    # runs. dateparser with DATE_ORDER="DMY" cannot read that shape, so try
    # ISO first and fall back to natural-language parsing.
    resolved: Optional[date] = None
    try:
        resolved = date.fromisoformat(text)
    except ValueError:
        parsed = dateparser.parse(
            text,
            settings={
                "PREFER_DATES_FROM": "future",
                "RELATIVE_BASE": datetime.combine(today, datetime.min.time()),
                "DATE_ORDER": "DMY",
            },
        )
        if parsed is None:
            return None
        resolved = parsed.date()

    if resolved < floor:
        return None
    if (resolved - floor).days > max_days_ahead:
        return None

    return resolved.isoformat()


class ActionValidateDepartureDate(Action):
    def name(self) -> Text:
        return "action_validate_departure_date"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        raw = tracker.get_slot("departure_date")
        result = normalise_date(raw)

        logger.info(
            "DEPARTURE DATE CHECK | raw=%r | today=%r | result=%r",
            raw,
            date.today(),
            result,
        )

        return [SlotSet("departure_date", result)]


class ActionValidateReturnDate(Action):
    def name(self) -> Text:
        return "action_validate_return_date"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        raw = tracker.get_slot("return_date")
        departure = tracker.get_slot("departure_date")

        floor = None
        if departure:
            try:
                floor = date.fromisoformat(departure)
            except ValueError:
                floor = None

        result = normalise_date(
            raw, not_before=floor, max_days_ahead=MAX_TRIP_DURATION_DAYS
        )

        logger.info(
            "RETURN DATE CHECK | raw=%r | departure=%r | result=%r",
            raw,
            departure,
            result,
        )

        return [SlotSet("return_date", result)]


def _resolve_into(field_name: str, tracker: Tracker) -> List[EventType]:
    """Shared resolution logic for origin and destination."""
    raw = tracker.get_slot(field_name)
    result = resolve_place(raw)

    logger.info(
        "PLACE RESOLVE | field=%s | raw=%r | status=%s | code=%r",
        field_name,
        raw,
        result.status,
        result.code,
    )

    if result.status == "resolved":
        return [
            SlotSet(f"{field_name}_code", result.code),
            SlotSet(f"{field_name}_display", result.display),
            SlotSet(f"{field_name}_options", None),
            SlotSet(f"{field_name}_options_text", None),
            SlotSet(f"{field_name}_choice", None),
        ]

    if result.status == "ambiguous":
        return [
            SlotSet(f"{field_name}_code", None),
            SlotSet(f"{field_name}_display", result.display),
            SlotSet(f"{field_name}_options", result.options),
            SlotSet(
                f"{field_name}_options_text",
                format_options(result.display, result.options),
            ),
            SlotSet(f"{field_name}_choice", None),
        ]

    # unknown or error
    return [
        SlotSet(f"{field_name}_code", None),
        SlotSet(f"{field_name}_options", None),
        SlotSet(f"{field_name}_options_text", None),
        SlotSet(field_name, None),
    ]


def _select_from(field_name: str, tracker: Tracker) -> List[EventType]:
    choice = tracker.get_slot(f"{field_name}_choice")
    options = tracker.get_slot(f"{field_name}_options") or []
    matched = match_choice(choice, options)

    logger.info(
        "AIRPORT CHOICE | field=%s | choice=%r | matched=%r",
        field_name,
        choice,
        matched,
    )

    if not matched:
        return [SlotSet(f"{field_name}_choice", None)]

    return [
        SlotSet(f"{field_name}_code", matched["iata_code"]),
        SlotSet(
            f"{field_name}_display",
            f"{matched['name']} ({matched['iata_code']})",
        ),
        SlotSet(f"{field_name}_options", None),
        SlotSet(f"{field_name}_options_text", None),
    ]


class ActionResolveOrigin(Action):
    def name(self) -> Text:
        return "action_resolve_origin"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _resolve_into("origin", tracker)


class ActionResolveDestination(Action):
    def name(self) -> Text:
        return "action_resolve_destination"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _resolve_into("destination", tracker)


class ActionSelectOriginAirport(Action):
    def name(self) -> Text:
        return "action_select_origin_airport"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _select_from("origin", tracker)


class ActionSelectDestinationAirport(Action):
    def name(self) -> Text:
        return "action_select_destination_airport"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _select_from("destination", tracker)


class ActionSearchFlights(Action):
    def name(self) -> Text:
        return "action_search_flights"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        origin = tracker.get_slot("origin_code")
        destination = tracker.get_slot("destination_code")
        departure = tracker.get_slot("departure_date")
        return_date = tracker.get_slot("return_date")
        trip_type = tracker.get_slot("trip_type")
        cabin = tracker.get_slot("cabin_class")
        raw_count = tracker.get_slot("passenger_count")

        # passenger_count is a float slot; Duffel needs an integer count.
        try:
            passengers = int(float(raw_count))
        except (TypeError, ValueError):
            passengers = 0

        if not all([origin, destination, departure, cabin]) or passengers < 1:
            logger.error("SEARCH ABORTED | incomplete slots")
            return [SlotSet("search_status", "error")]

        slices = build_slices(
            origin,
            destination,
            departure,
            return_date if trip_type == "return" else None,
        )

        try:
            offers = create_offer_request(slices, passengers, cabin)
        except DuffelError as exc:
            logger.error("SEARCH FAILED | %s", exc)
            return [SlotSet("search_status", "error")]

        if not offers:
            logger.info("SEARCH EMPTY | no offers returned")
            return [SlotSet("search_status", "empty"), SlotSet("offers", [])]

        # Cheapest first. Deterministic ordering, not model-ranked.
        offers.sort(key=lambda o: float(o.get("total_amount") or 1e12))
        summaries = [summarise_offer(o) for o in offers[:MAX_OFFERS_STORED]]

        logger.info(
            "SEARCH OK | returned=%d | stored=%d | cheapest=%s %s | expires=%s",
            len(offers),
            len(summaries),
            summaries[0]["total_amount"],
            summaries[0]["total_currency"],
            summaries[0]["expires_at"],
        )

        return [
            SlotSet("search_status", "ok"),
            SlotSet("offers", summaries),
            SlotSet("offer_count", float(len(offers))),
        ]


class ActionPresentOffers(Action):
    def name(self) -> Text:
        return "action_present_offers"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        offers = tracker.get_slot("offers") or []

        if not offers:
            return [SlotSet("offers_text", None)]

        shortlist = offers[:MAX_OFFERS_SHOWN]
        remaining = minutes_until_expiry(shortlist[0])

        logger.info(
            "PRESENT OFFERS | shown=%d | of=%d | expires_in_min=%s",
            len(shortlist),
            len(offers),
            remaining,
        )

        return [
            SlotSet("offers_text", format_offer_list(shortlist)),
            SlotSet("offers_shown", float(len(shortlist))),
            SlotSet("offer_choice", None),
            SlotSet("selected_offer_id", None),
        ]


class ActionSelectOffer(Action):
    def name(self) -> Text:
        return "action_select_offer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[EventType]:
        choice = tracker.get_slot("offer_choice")
        offers = tracker.get_slot("offers") or []
        shortlist = offers[:MAX_OFFERS_SHOWN]
        index = None
        if choice is not None:
            text = str(choice).strip().lower()

            # Ordinal and preference words a user may reasonably use.
            # The shortlist is already sorted cheapest-first, so "cheapest"
            # is position 1 by construction, not by model judgement.
            word_positions = {
                "cheapest": 1,
                "first": 1,
                "1st": 1,
                "second": 2,
                "2nd": 2,
                "third": 3,
                "3rd": 3,
                "fourth": 4,
                "4th": 4,
                "fifth": 5,
                "5th": 5,
            }

        for word, position in word_positions.items():
            if word in text:
                index = position
                break

        if index is None:
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                index = int(digits)

        if index is None or not (1 <= index <= len(shortlist)):
            logger.info("OFFER CHOICE INVALID | choice=%r", choice)
            return [SlotSet("offer_choice", None)]

        chosen = shortlist[index - 1]
        remaining = minutes_until_expiry(chosen)

        logger.info(
            "OFFER SELECTED | index=%d | id=%s | %s %s | expires_in_min=%s",
            index,
            chosen["id"],
            chosen["total_currency"],
            chosen["total_amount"],
            remaining,
        )

        # Offer expiry (F7): an offer that has already lapsed cannot be booked.
        if remaining is not None and remaining <= 0:
            return [
                SlotSet("offer_choice", None),
                SlotSet("selected_offer_id", None),
                SlotSet("offer_expired", True),
            ]

        return [
            SlotSet("selected_offer_id", chosen["id"]),
            SlotSet("selected_offer_text", format_offer(index, chosen)),
            SlotSet("offer_expired", False),
        ]
MIN_ADULT_AGE_YEARS = 18
MAX_AGE_YEARS = 120


def validate_dob(raw: Any) -> Optional[str]:
    """Date of birth must be a real past date giving a plausible adult age.

    The MVP books adults only, so anyone under 18 is rejected rather than
    silently reclassified as a child fare (see Chapter 5).
    """
    if raw in (None, ""):
        return None

    text = str(raw).strip()
    born: Optional[date] = None
    try:
        born = date.fromisoformat(text)
    except ValueError:
        parsed = dateparser.parse(
            text,
            settings={"PREFER_DATES_FROM": "past", "DATE_ORDER": "DMY"},
        )
        if parsed is None:
            return None
        born = parsed.date()

    today = date.today()
    if born >= today:
        return None

    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    if age < MIN_ADULT_AGE_YEARS or age > MAX_AGE_YEARS:
        return None

    return born.isoformat()


def _validated(slot: str, value: Optional[str]) -> List[EventType]:
    logger.info("FIELD CHECK | slot=%s | accepted=%s", slot, value is not None)
    return [SlotSet(slot, value)]


class ActionValidateGivenName(Action):
    def name(self) -> Text:
        return "action_validate_given_name"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "given_name", normalise_name(tracker.get_slot("given_name"))
        )


class ActionValidateFamilyName(Action):
    def name(self) -> Text:
        return "action_validate_family_name"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "family_name", normalise_name(tracker.get_slot("family_name"))
        )


class ActionValidateDob(Action):
    def name(self) -> Text:
        return "action_validate_dob"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated("born_on", validate_dob(tracker.get_slot("born_on")))


class ActionValidateTitle(Action):
    def name(self) -> Text:
        return "action_validate_title"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "passenger_title", normalise_title(tracker.get_slot("passenger_title"))
        )


class ActionValidateGender(Action):
    def name(self) -> Text:
        return "action_validate_gender"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "passenger_gender", normalise_gender(tracker.get_slot("passenger_gender"))
        )


class ActionValidateEmail(Action):
    def name(self) -> Text:
        return "action_validate_email"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "passenger_email", normalise_email(tracker.get_slot("passenger_email"))
        )


class ActionValidatePhone(Action):
    def name(self) -> Text:
        return "action_validate_phone"

    def run(self, dispatcher, tracker, domain) -> List[EventType]:
        return _validated(
            "passenger_phone", normalise_phone(tracker.get_slot("passenger_phone"))
        )