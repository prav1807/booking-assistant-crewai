import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
REQUEST_TIMEOUT_SECONDS = 20


class SupabaseError(RuntimeError):
    """Raised when persistence fails. Never swallowed silently — a booking
    that is not recorded cannot be audited (NFR9)."""


def _headers(prefer: Optional[str] = None) -> Dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise SupabaseError("SUPABASE_URL or SUPABASE_SERVICE_KEY is not set.")
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _post(table: str, rows: Any, prefer: str = "return=representation") -> List[Dict]:
    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_headers(prefer),
            json=rows,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SupabaseError(f"Could not reach Supabase: {exc}") from exc

    if response.status_code not in (200, 201):
        raise SupabaseError(
            f"Supabase {table} insert returned "
            f"{response.status_code}: {response.text[:300]}"
        )

    return response.json() if response.text else []


def _patch(table: str, filters: str, values: Dict[str, Any]) -> List[Dict]:
    try:
        response = requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}?{filters}",
            headers=_headers("return=representation"),
            json=values,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SupabaseError(f"Could not reach Supabase: {exc}") from exc

    if response.status_code not in (200, 204):
        raise SupabaseError(
            f"Supabase {table} update returned "
            f"{response.status_code}: {response.text[:300]}"
        )

    return response.json() if response.text else []


def find_booking_by_key(idempotency_key: str) -> Optional[Dict[str, Any]]:
    """Return an existing booking for this key, if one was already created.

    This is the read half of the idempotency guarantee (FR14): before
    retrying a commitment, check whether it already succeeded.
    """
    try:
        response = requests.get(
            f"{SUPABASE_URL}/rest/v1/bookings",
            headers=_headers(),
            params={"idempotency_key": f"eq.{idempotency_key}", "select": "*"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise SupabaseError(f"Could not reach Supabase: {exc}") from exc

    if response.status_code != 200:
        raise SupabaseError(
            f"Supabase lookup returned {response.status_code}: {response.text[:300]}"
        )

    rows = response.json()
    return rows[0] if rows else None


def create_pending_booking(
    sender_id: str,
    idempotency_key: str,
    order_type: str,
    search: Dict[str, Any],
) -> Dict[str, Any]:
    """Record the intent to book BEFORE calling the provider.

    Writing first means a provider call that times out still leaves a
    trace, so the retry path can reconcile rather than double-book (F8).
    """
    row = {
        "sender_id": sender_id,
        "idempotency_key": idempotency_key,
        "order_type": order_type,
        "status": "pending",
        "origin_code": search.get("origin_code"),
        "destination_code": search.get("destination_code"),
        "departure_date": search.get("departure_date"),
        "return_date": search.get("return_date"),
        "cabin_class": search.get("cabin_class"),
    }
    created = _post("bookings", row)
    logger.info("BOOKING PENDING | key=%s | id=%s", idempotency_key,
                created[0]["id"] if created else None)
    return created[0]


def confirm_booking(
    booking_id: str,
    duffel_order_id: str,
    booking_reference: str,
    total_amount: str,
    total_currency: str,
    payment_required_by: Optional[str],
) -> None:
    _patch(
        "bookings",
        f"id=eq.{booking_id}",
        {
            "duffel_order_id": duffel_order_id,
            "booking_reference": booking_reference,
            "total_amount": total_amount,
            "total_currency": total_currency,
            "payment_required_by": payment_required_by,
            "status": "confirmed",
            "updated_at": "now()",
        },
    )
    logger.info("BOOKING CONFIRMED | id=%s | ref=%s", booking_id, booking_reference)


def fail_booking(booking_id: str, reason: str) -> None:
    _patch("bookings", f"id=eq.{booking_id}",
           {"status": "failed", "updated_at": "now()"})
    logger.error("BOOKING FAILED | id=%s | %s", booking_id, reason)


def save_passenger(booking_id: str, passenger: Dict[str, Any]) -> None:
    _post("passengers", {**passenger, "booking_id": booking_id},
          prefer="return=minimal")


def save_offer_snapshot(booking_id: Optional[str], offer: Dict[str, Any]) -> None:
    """Persist the offer exactly as priced at commitment (FR15)."""
    _post(
        "offer_snapshots",
        {
            "booking_id": booking_id,
            "duffel_offer_id": offer.get("id", ""),
            "snapshot": offer,
        },
        prefer="return=minimal",
    )


def write_audit(
    sender_id: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    model_name: Optional[str] = None,
) -> None:
    """Append to the audit log. Failures are logged but never raised —
    audit writing must not itself break a booking."""
    try:
        _post(
            "audit_log",
            {
                "sender_id": sender_id,
                "event_type": event_type,
                "payload": payload or {},
                "model_name": model_name,
            },
            prefer="return=minimal",
        )
    except SupabaseError as exc:
        logger.error("AUDIT WRITE FAILED | %s | %s", event_type, exc)