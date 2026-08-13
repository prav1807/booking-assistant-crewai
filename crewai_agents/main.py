"""Entry point — run the Flight Search CrewAI agent from the command line."""

import sys
import os

# Fix Windows console encoding for Unicode characters (arrows, emojis)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai_agents.flight_search_crew import run_flight_search


def main():
    print("=" * 60)
    print(" Flight Search Agent (CrewAI + Ollama + Duffel)")
    print("=" * 60)

    if len(sys.argv) > 1:
        user_request = " ".join(sys.argv[1:])
    else:
        user_request = input(
            "\nDescribe your flight search:\n"
            "(e.g. 'Flights from Mauritius to London on 20 September 2026, "
            "economy, 1 passenger')\n\n> "
        )

    if not user_request.strip():
        print("No request provided. Exiting.")
        return

    print(f"\nSearching for: {user_request}\n")
    print("-" * 60)

    result = run_flight_search(user_request)

    print("\n" + "=" * 60)
    print(" RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
