"""BookingReviewCrew — pre-booking payload verification (Role 5).

Invoked immediately before order creation. The BookingVerificationAgent
re-checks the assembled order payload, passenger details, selected offer
and fare conditions against the collected conversation state.

This is the last point at which an inconsistency can be caught without
financial consequence.
"""

from typing import Any, Dict

from crewai import Agent, Crew, Process, Task

from .llm_config import build_llm
from .tools.booking_review_tools import PayloadVerificationTool


def create_booking_verification_agent() -> Agent:
    return Agent(
        role="Booking Verification Specialist",
        goal=(
            "Verify the assembled booking order payload before it is submitted. "
            "Check that passenger details, offer selection, route, and dates all "
            "match the conversation state. Flag any inconsistency."
        ),
        backstory=(
            "You are a meticulous airline booking auditor. Before any order is "
            "submitted to the airline, you verify every detail matches what was "
            "discussed. You never let a mismatched or expired booking through."
        ),
        tools=[PayloadVerificationTool()],
        llm=build_llm(),
        verbose=True,
        max_iter=3,
    )


def run_booking_review(
    order_payload: Dict[str, Any],
    passenger_details: Dict[str, Any],
    selected_offer: Dict[str, Any],
    conversation_slots: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the BookingReviewCrew to verify the order before submission.

    Args:
        order_payload: The Duffel order request body.
        passenger_details: {given_name, family_name, born_on, title, gender, email, phone}
        selected_offer: {id, total_amount, total_currency, expires_at}
        conversation_slots: All relevant conversation slots for cross-checking.

    Returns:
        {verified: bool, issues: list, reason: str}
    """
    # Fast deterministic check first (no LLM needed)
    tool = PayloadVerificationTool()
    result = tool._run(
        order_payload=order_payload,
        passenger_details=passenger_details,
        selected_offer=selected_offer,
        conversation_slots=conversation_slots,
    )

    if "VERIFIED" in result:
        return {
            "verified": True,
            "issues": [],
            "reason": "All checks passed.",
        }

    # Parse issues
    issues = []
    if "ISSUES_FOUND" in result:
        for line in result.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                issues.append(line[2:])

    # If issues are clear-cut (MISSING/MISMATCH/EXPIRED), no LLM needed
    if issues:
        return {
            "verified": False,
            "issues": issues,
            "reason": "; ".join(issues),
        }

    # Fallback: use the LLM agent for ambiguous cases
    agent = create_booking_verification_agent()
    task = Task(
        description=(
            f"Verify this booking payload:\n"
            f"Order: {order_payload}\n"
            f"Passenger: {passenger_details}\n"
            f"Offer: {selected_offer}\n"
            f"Conversation slots: {conversation_slots}\n\n"
            "Use the verify_booking_payload tool to check for inconsistencies."
        ),
        expected_output="A verification verdict: either VERIFIED or a list of issues.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    crew_result = crew.kickoff()
    raw = str(crew_result)

    verified = "VERIFIED" in raw and "ISSUES" not in raw
    return {
        "verified": verified,
        "issues": issues,
        "reason": raw[:500] if not verified else "All checks passed.",
    }
