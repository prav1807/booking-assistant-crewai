"""Quick test for the ValidationCrew."""
import sys
import os
import json

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai_agents.validation_crew import run_validation_crew


def test_valid_request():
    print("=" * 60)
    print(" Test 1: Valid Booking Request")
    print("=" * 60)
    slots = {
        "origin_code": "LHR",
        "destination_code": "JFK",
        "departure_date": "2026-10-01",
        "return_date": "2026-10-15",
        "passenger_count": 2,
        "cabin_class": "business",
        "trip_type": "return",
    }
    print(f"Slots: {json.dumps(slots, indent=2)}\n")
    verdict = run_validation_crew(slots)
    print_verdict(verdict)
    return verdict


def test_invalid_request():
    print("\n" + "=" * 60)
    print(" Test 2: Invalid Booking Request (past date, same airports)")
    print("=" * 60)
    slots = {
        "origin_code": "LHR",
        "destination_code": "LHR",
        "departure_date": "2020-01-01",
        "return_date": None,
        "passenger_count": 0,
        "cabin_class": "luxury",
        "trip_type": "return",
    }
    print(f"Slots: {json.dumps(slots, indent=2)}\n")
    verdict = run_validation_crew(slots)
    print_verdict(verdict)
    return verdict


def print_verdict(verdict):
    print("\n" + "-" * 40)
    print("VERDICT:")
    print(f"  Valid: {verdict['valid']}")
    print(f"  Errors: {verdict['errors']}")
    print(f"  Warnings: {verdict['warnings']}")
    if verdict["duffel_request"]:
        print(f"  Duffel Request: {json.dumps(verdict['duffel_request'], indent=4)}")
    print("-" * 40)


if __name__ == "__main__":
    v1 = test_valid_request()
    v2 = test_invalid_request()

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (valid): {'PASS' if v1['valid'] else 'FAIL'}")
    print(f"  Test 2 (invalid): {'PASS' if not v2['valid'] else 'FAIL'}")
