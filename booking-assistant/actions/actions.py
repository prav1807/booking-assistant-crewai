import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Text

import dateparser
from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher

logger = logging.getLogger(__name__)

MAX_BOOKING_HORIZON_DAYS = 365


def normalise_date(raw: Any, not_before: Optional[date] = None) -> Optional[str]:
    """Return an ISO date string, or None if the input is unusable.

    None is the signal for 'invalid' — the flow branches on the slot being
    empty, so no separate validity flag is needed.
    """
    if raw in (None, ""):
        return None

    today = date.today()
    floor = not_before or today

    parsed = dateparser.parse(
        str(raw),
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
    if (resolved - today).days > MAX_BOOKING_HORIZON_DAYS:
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

        logger.warning(
            "DEPARTURE DATE CHECK | raw=%r | today=%r | result=%r",
            raw, date.today(), result,
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

        result = normalise_date(raw, not_before=floor)

        logger.warning(
            "RETURN DATE CHECK | raw=%r | departure_slot=%r | floor=%r | today=%r | result=%r",
            raw, departure, floor, date.today(), result,
        )

        return [SlotSet("return_date", result)]