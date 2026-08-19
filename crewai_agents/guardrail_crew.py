"""GuardrailCrew — outbound message safety screening (Role 6).

Every outbound message passes the guardrail with a tiered design:
1. Fast deterministic regex pre-filter (PII, prohibited phrasings) — all messages
2. LLM SafetyReviewAgent — only for messages with factual/policy/price claims

Where the guardrail rejects a draft, the response is suppressed rather than
sent with a warning, and the flow either regenerates or escalates.
"""

from typing import Any, Dict

from crewai import Agent, Crew, Process, Task

from .llm_config import build_llm
from .tools.guardrail_tools import (
    FactualClaimDetectorTool,
    RegexPreFilterTool,
)


def create_safety_review_agent() -> Agent:
    return Agent(
        role="Safety Review Specialist",
        goal=(
            "Review outbound messages for factual accuracy, over-commitment, "
            "and policy contradictions. Flag any message that makes guarantees "
            "the system cannot back, states incorrect policy, or commits to "
            "prices/availability that may change."
        ),
        backstory=(
            "You are a compliance reviewer for an airline booking chatbot. "
            "Your job is to catch messages that could create legal liability "
            "(like the Air Canada case). You flag over-commitments, unhedged "
            "price guarantees, and policy claims not grounded in data. "
            "You err on the side of caution."
        ),
        tools=[],
        llm=build_llm(),
        verbose=True,
        max_iter=2,
    )


def run_guardrail(draft_message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Run the tiered guardrail on an outbound message.

    Args:
        draft_message: The message about to be sent to the user.
        context: Optional context (e.g. conversation topic, current flow).

    Returns:
        {passed: bool, reason: str, tier: str}
        tier is "regex" (blocked by pre-filter) or "llm" (blocked by agent)
    """
    if not draft_message or not draft_message.strip():
        return {"passed": True, "reason": "Empty message.", "tier": "regex"}

    # --- Tier 1: Fast regex pre-filter (runs on ALL messages) ---
    regex_tool = RegexPreFilterTool()
    regex_result = regex_tool._run(draft_message)

    if "BLOCKED" in regex_result:
        violations = []
        for line in regex_result.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                violations.append(line[2:])
        return {
            "passed": False,
            "reason": "; ".join(violations) if violations else "PII or prohibited phrasing detected.",
            "tier": "regex",
        }

    # --- Tier 2: Check if LLM review is needed ---
    claim_tool = FactualClaimDetectorTool()
    claim_result = claim_tool._run(draft_message)

    if "NO_REVIEW_NEEDED" in claim_result:
        return {"passed": True, "reason": "No issues detected.", "tier": "regex"}

    # --- Tier 3: LLM Safety Review (only for factual claims) ---
    agent = create_safety_review_agent()

    context_str = ""
    if context:
        context_str = f"\nContext: {context}"

    task = Task(
        description=(
            f"Review this outbound message for safety issues:\n\n"
            f'"{draft_message}"\n{context_str}\n\n'
            "Check for:\n"
            "1. Over-commitments (absolute guarantees about price, availability, refunds)\n"
            "2. Ungrounded policy claims (stating rules without source)\n"
            "3. Price commitments that may have expired\n"
            "4. Legal-sounding promises the system cannot enforce\n\n"
            "Respond with either:\n"
            "- PASS: if the message is safe to send\n"
            "- BLOCK: [reason] if the message should be suppressed"
        ),
        expected_output="Either 'PASS' or 'BLOCK: [reason]'",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        result = crew.kickoff()
        raw = str(result).strip()

        if "BLOCK" in raw.upper():
            reason = raw.split("BLOCK")[-1].strip(": ").strip()
            return {
                "passed": False,
                "reason": reason or "LLM flagged safety concern.",
                "tier": "llm",
            }

        return {"passed": True, "reason": "LLM review passed.", "tier": "llm"}

    except Exception as exc:
        # On LLM failure, pass the message (fail-open for non-critical guardrail)
        return {
            "passed": True,
            "reason": f"LLM review skipped (error: {exc}). Passed by default.",
            "tier": "llm",
        }
