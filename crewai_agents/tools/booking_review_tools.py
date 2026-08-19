"""Tools for the BookingReviewCrew — pre-booking payload verification."""

from typing import Any, Dict, List, Optional, Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class PayloadVerificationInput(BaseModel):
    order_payload: Dict[str, Any] = Field(
        description="The assembled Duffel order payload to verify"
    )
    passenger_details: Dict[str, Any] = Field(
        description="Collected passenger details (given_name, family_name, born_on, title, gender, email, phone)"
    )
    selected_offer: Dict[str, Any] = Field(
        description="The selected offer (id, total_amount, total_currency, expires_at)"
    )
    conversation_slots: Dict[str, Any] = Field(
        description="All conversation slots for cross-checking"
    )


class PayloadVerificationTool(BaseTool):
    name: str = "verify_booking_payload"
    description: str = (
        "Verify the assembled booking order payload against the conversation state. "
        "Checks that passenger details match collected slots, the offer is still valid, "
        "and all required fields are present. Returns a list of issues found."
    )
    args_schema: Type[BaseModel] = PayloadVerificationInput

    def _run(
        self,
        order_payload: Dict[str, Any],
        passenger_details: Dict[str, Any],
        selected_offer: Dict[str, Any],
        conversation_slots: Dict[str, Any],
    ) -> str:
        issues = []

        # 1. Check required passenger fields
        required_passenger_fields = [
            "given_name", "family_name", "born_on", "title", "gender", "email", "phone"
        ]
        for field in required_passenger_fields:
            if not passenger_details.get(field):
                issues.append(f"MISSING: Passenger field '{field}' is empty.")

        # 2. Cross-check passenger details against conversation slots
        slot_field_map = {
            "given_name": "given_name",
            "family_name": "family_name",
            "born_on": "born_on",
            "title": "passenger_title",
            "gender": "passenger_gender",
            "email": "passenger_email",
            "phone": "passenger_phone",
        }
        for pax_field, slot_name in slot_field_map.items():
            pax_value = str(passenger_details.get(pax_field, "")).strip().lower()
            slot_value = str(conversation_slots.get(slot_name, "")).strip().lower()
            if pax_value and slot_value and pax_value != slot_value:
                issues.append(
                    f"MISMATCH: Passenger '{pax_field}' is '{passenger_details.get(pax_field)}' "
                    f"but conversation slot '{slot_name}' has '{conversation_slots.get(slot_name)}'."
                )

        # 3. Check offer validity
        offer_id = selected_offer.get("id")
        if not offer_id:
            issues.append("MISSING: No offer ID in selected_offer.")

        slot_offer_id = conversation_slots.get("selected_offer_id")
        if offer_id and slot_offer_id and offer_id != slot_offer_id:
            issues.append(
                f"MISMATCH: Order uses offer '{offer_id}' but selected_offer_id slot is '{slot_offer_id}'."
            )

        # 4. Check offer expiry
        from datetime import datetime, timezone
        expires_at = selected_offer.get("expires_at")
        if expires_at:
            try:
                expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                if expires < datetime.now(timezone.utc):
                    issues.append("EXPIRED: The selected offer has already expired.")
            except ValueError:
                issues.append(f"INVALID: Cannot parse offer expiry '{expires_at}'.")

        # 5. Check route consistency
        origin = conversation_slots.get("origin_code")
        destination = conversation_slots.get("destination_code")
        slices = order_payload.get("data", {}).get("slices", [])
        if slices:
            first_slice = slices[0]
            if origin and first_slice.get("origin") != origin:
                issues.append(
                    f"MISMATCH: Order origin '{first_slice.get('origin')}' != slot origin '{origin}'."
                )
            if destination and first_slice.get("destination") != destination:
                issues.append(
                    f"MISMATCH: Order destination '{first_slice.get('destination')}' != slot destination '{destination}'."
                )

        # 6. Check passenger count
        passengers_in_payload = len(order_payload.get("data", {}).get("passengers", []))
        slot_count = conversation_slots.get("passenger_count")
        if slot_count:
            expected = int(float(slot_count))
            if passengers_in_payload != expected:
                issues.append(
                    f"MISMATCH: Order has {passengers_in_payload} passenger(s) but slot says {expected}."
                )

        if not issues:
            return "VERIFIED: All checks passed. The booking payload is consistent with the conversation state."

        return "ISSUES_FOUND:\n" + "\n".join(f"  - {issue}" for issue in issues)
