"""Flight Search Crew — searches and compares flights using Duffel API."""

from crewai import Agent, Crew, Process, Task

from .llm_config import build_llm
from .tools.flight_tools import AirportLookupTool, FlightSearchTool


def create_flight_search_agent() -> Agent:
    """Create the flight search specialist agent."""
    return Agent(
        role="Flight Search Specialist",
        goal=(
            "Find available flights for the user's trip. "
            "Look up airport codes if needed, search for flights, "
            "then present results ranked by value."
        ),
        backstory=(
            "You are an experienced travel agent AI. You know airline routes "
            "and airport codes. Always look up IATA codes first if given city "
            "names, then search for flights."
        ),
        tools=[AirportLookupTool(), FlightSearchTool()],
        llm=build_llm(),
        verbose=True,
        max_iter=5,
    )


def create_flight_search_task(agent: Agent, user_request: str) -> Task:
    """Create a flight search task from the user's natural language request."""
    return Task(
        description=(
            f"User request: {user_request}\n\n"
            "Steps:\n"
            "1. If the user provided city names instead of airport codes, use the "
            "   lookup_airport tool to find the correct IATA codes.\n"
            "2. Use the search_flights tool with the correct IATA codes, dates, "
            "   passenger count, and cabin class.\n"
            "3. Analyse the results: compare prices, flight durations, and stops.\n"
            "4. Present a clear summary of the top options and recommend the best "
            "   value flight."
        ),
        expected_output=(
            "A structured summary of available flights including:\n"
            "- Top 5 options ranked by value (price vs duration vs stops)\n"
            "- Each option showing: airline, price, route, departure/arrival times, "
            "  duration, and number of stops\n"
            "- A final recommendation with reasoning"
        ),
        agent=agent,
    )


def run_flight_search(user_request: str) -> str:
    """Run the flight search crew with a user's natural language request.

    Args:
        user_request: e.g. "Find me flights from Mauritius to London on 15 September
                      2026, economy class, 2 passengers"

    Returns:
        The crew's analysis and recommendations as a string.
    """
    agent = create_flight_search_agent()
    task = create_flight_search_task(agent, user_request)

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
