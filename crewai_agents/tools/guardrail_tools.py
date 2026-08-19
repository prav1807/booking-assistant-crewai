"""Tools for the GuardrailCrew — PII detection and safety review."""

import re
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# --- PII and prohibited phrasing patterns ---

PII_PATTERNS = {
    "credit_card": re.compile(
        r"\b(?:\d[ -]*?){13,19}\b"
    ),
    "passport_number": re.compile(
        r"\b[A-Z]{1,2}\d{6,9}\b"
    ),
    "email_in_prose": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    ),
    "phone_in_prose": re.compile(
        r"\+\d{1,3}[-.\s]?\d{6,14}\b"
    ),
    "ssn": re.compile(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ),
}

PROHIBITED_PHRASES = [
    r"\bguarantee(?:d|s)?\b.*\b(?:price|fare|seat|availability)\b",
    r"\bpromise\b.*\b(?:refund|compensation|price)\b",
    r"\bwill\s+(?:definitely|certainly|always)\b",
    r"\byou\s+(?:are|will be)\s+entitled\b",
    r"\blegal(?:ly)?\s+(?:obligat|requir|entitl)\b",
    r"\bno\s+(?:extra|additional)\s+(?:charge|fee|cost)\s+(?:ever|guaranteed)\b",
]

FACTUAL_CLAIM_INDICATORS = [
    r"\b(?:GBP|USD|EUR|£|\$|€)\s*\d+",
    r"\b\d+\s*(?:kg|lb|cm|inch)",
    r"\b(?:refund|cancel|change|baggag|visa|transit)\b",
    r"\b(?:EU\s+Regulation|IATA|14\s+CFR|DOT)\b",
    r"\b(?:guaranteed|confirmed|entitled|eligible)\b",
    r"\b(?:free\s+of\s+charge|no\s+fee|complimentary)\b",
]


class RegexPreFilterInput(BaseModel):
    message: str = Field(description="The outbound message draft to screen")


class RegexPreFilterTool(BaseTool):
    name: str = "regex_pre_filter"
    description: str = (
        "Screen an outbound message for PII (credit cards, passport numbers, "
        "emails, phone numbers) and prohibited phrasings (absolute guarantees, "
        "legal promises). Fast deterministic check — no LLM needed."
    )
    args_schema: Type[BaseModel] = RegexPreFilterInput

    def _run(self, message: str) -> str:
        if not message:
            return "PASS: Empty message."

        violations = []

        # Check PII
        for pii_type, pattern in PII_PATTERNS.items():
            matches = pattern.findall(message)
            if matches:
                violations.append(f"PII_DETECTED ({pii_type}): Found {len(matches)} match(es).")

        # Check prohibited phrases
        for i, pattern_str in enumerate(PROHIBITED_PHRASES):
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(message):
                violations.append(f"PROHIBITED_PHRASING (rule {i+1}): Over-commitment detected.")

        if violations:
            return "BLOCKED:\n" + "\n".join(f"  - {v}" for v in violations)

        return "PASS: No PII or prohibited phrasings detected."


class FactualClaimDetectorInput(BaseModel):
    message: str = Field(description="The outbound message to check for factual claims")


class FactualClaimDetectorTool(BaseTool):
    name: str = "detect_factual_claims"
    description: str = (
        "Detect whether an outbound message contains factual, policy, or price "
        "claims that require deeper LLM review. Returns whether LLM review is needed."
    )
    args_schema: Type[BaseModel] = FactualClaimDetectorInput

    def _run(self, message: str) -> str:
        if not message:
            return "NO_REVIEW_NEEDED: Empty message."

        triggers = []
        for pattern_str in FACTUAL_CLAIM_INDICATORS:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            if pattern.search(message):
                triggers.append(pattern_str)

        if triggers:
            return f"REVIEW_NEEDED: Message contains {len(triggers)} factual claim indicator(s)."

        return "NO_REVIEW_NEEDED: No factual claims detected."
