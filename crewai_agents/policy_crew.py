"""PolicyCrew — answers policy and eligibility questions from grounded sources.

Two agents per the report (Section 4.5.3):
  1. PolicyRetrievalAgent — searches the Qdrant-indexed policy corpus
     (IATA baggage, dangerous goods, EU 1107/2006, US 14 CFR 382, fare rules)
  2. EligibilityAgent — resolves visa/entry questions via the passport-index
     structured lookup (deterministic, not retrieval)

Returns grounded answers with source citations, never fabricated information.
"""

from typing import Any, Dict

from crewai import Agent, Crew, Process, Task

from .llm_config import build_llm
from .tools.policy_tools import PolicyRetrievalTool, VisaEligibilityTool


def create_policy_retrieval_agent() -> Agent:
    return Agent(
        role="Policy Retrieval Specialist",
        goal=(
            "Answer traveller questions about airline policies by searching "
            "the policy knowledge base. Every answer must be grounded in a "
            "retrieved passage and cite its source. Never fabricate policy "
            "information."
        ),
        backstory=(
            "You are an airline policy expert with access to authoritative "
            "documents covering baggage rules, dangerous goods regulations, "
            "disability assistance provisions, and fare conditions. You always "
            "search the knowledge base before answering and cite the source "
            "of each fact."
        ),
        tools=[PolicyRetrievalTool()],
        llm=build_llm(),
        verbose=True,
        max_iter=3,
    )


def create_eligibility_agent() -> Agent:
    return Agent(
        role="Visa and Entry Eligibility Specialist",
        goal=(
            "Determine visa and entry requirements for a traveller based on "
            "their passport nationality and destination country. Use the "
            "structured passport-index lookup for deterministic answers."
        ),
        backstory=(
            "You are an immigration and travel eligibility specialist. You "
            "resolve visa questions using the official passport-index dataset, "
            "which provides definitive visa-free, visa-required, or eVisa "
            "status for every passport-destination pair. You never guess — "
            "you always look up the data."
        ),
        tools=[VisaEligibilityTool()],
        llm=build_llm(),
        verbose=True,
        max_iter=3,
    )


def _is_visa_question(question: str) -> bool:
    """Heuristic to detect if a question is about visa/entry eligibility."""
    keywords = ("visa", "entry", "passport", "travel to", "need a visa",
                "visa-free", "transit", "eligibility", "immigration")
    q = question.lower()
    return any(kw in q for kw in keywords)


def run_policy_crew(
    question: str,
    passport_country: str = None,
    destination_country: str = None,
) -> Dict[str, Any]:
    """Run the PolicyCrew to answer a policy or eligibility question.

    Args:
        question: The traveller's natural language question.
        passport_country: Passport nationality (for visa questions).
        destination_country: Destination country (for visa questions).

    Returns:
        {
            "answered": bool,
            "answer": str,
            "sources": list[str],
            "question_type": "policy" | "eligibility",
        }
    """
    is_visa = _is_visa_question(question) and passport_country and destination_country

    if is_visa:
        return _run_eligibility(question, passport_country, destination_country)
    else:
        return _run_policy_retrieval(question)


def _run_policy_retrieval(question: str) -> Dict[str, Any]:
    """Answer a policy question using RAG."""
    agent = create_policy_retrieval_agent()

    task = Task(
        description=(
            f"Traveller question: \"{question}\"\n\n"
            "Steps:\n"
            "1. Search the policy knowledge base for relevant passages.\n"
            "2. Read the retrieved passages carefully.\n"
            "3. Compose a clear, concise answer using ONLY information from "
            "   the retrieved passages.\n"
            "4. Cite the source of each fact (e.g. 'According to IATA Passenger "
            "   Baggage Rules, ...').\n"
            "5. If no relevant passages are found, say so honestly — do NOT "
            "   make up an answer."
        ),
        expected_output=(
            "A grounded answer to the traveller's question with source citations. "
            "If the knowledge base has no relevant information, state that clearly."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    raw = str(result)

    # Collect sources from output
    sources = []
    for marker in ["IATA", "EU Regulation", "US 14 CFR", "General airline fare"]:
        if marker in raw:
            sources.append(marker)

    return {
        "answered": bool(raw.strip()) and "NO_RESULTS" not in raw,
        "answer": raw.strip(),
        "sources": sources,
        "question_type": "policy",
    }


def _run_eligibility(
    question: str, passport_country: str, destination_country: str
) -> Dict[str, Any]:
    """Answer a visa/entry question using structured lookup."""
    agent = create_eligibility_agent()

    task = Task(
        description=(
            f"Traveller question: \"{question}\"\n"
            f"Passport country: {passport_country}\n"
            f"Destination country: {destination_country}\n\n"
            "Use the check_visa_eligibility tool to look up the entry requirement "
            "for this passport-destination pair. Report the result clearly."
        ),
        expected_output=(
            "A clear statement of the visa/entry requirement with the data source cited."
        ),
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    raw = str(result)

    return {
        "answered": bool(raw.strip()) and "NO_DATA" not in raw,
        "answer": raw.strip(),
        "sources": ["passport-index dataset (Ilyankou, 2025)"],
        "question_type": "eligibility",
    }
