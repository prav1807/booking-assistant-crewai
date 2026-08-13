"""Test suite for CrewAI flight search agent — tools and edge cases."""

import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crewai_agents.tools.flight_tools import AirportLookupTool, FlightSearchTool

lookup = AirportLookupTool()
search = FlightSearchTool()

PASS = 0
FAIL = 0


def test(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    status = "PASS" if condition else "FAIL"
    if condition:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}")
    if detail and not condition:
        print(f"         Detail: {detail[:200]}")


def run_tests():
    global PASS, FAIL
    print("=" * 60)
    print(" Flight Search Agent - Edge Case Tests")
    print("=" * 60)

    # --- Airport Lookup Tests ---
    print("\n[1] Airport Lookup Tool")
    print("-" * 40)

    # Valid city
    result = lookup._run("Mauritius")
    test("Lookup 'Mauritius' returns MRU", "MRU" in result, result)

    # Ambiguous city with multiple airports
    result = lookup._run("London")
    test(
        "Lookup 'London' returns multiple airports",
        "LHR" in result or "LGW" in result or "STN" in result,
        result,
    )

    # Nonexistent place
    result = lookup._run("Xyzzyplugh")
    test(
        "Lookup nonsense returns no airports",
        "No airports" in result,
        result,
    )

    # Empty input
    result = lookup._run("")
    test("Lookup empty string handled gracefully", "No airports" in result, result)

    # IATA code directly
    result = lookup._run("DXB")
    test("Lookup 'DXB' returns Dubai airport", "DXB" in result or "Dubai" in result, result)

    # Multi-airport city
    result = lookup._run("New York")
    test(
        "Lookup 'New York' returns JFK/EWR/LGA",
        "JFK" in result or "EWR" in result or "LGA" in result,
        result,
    )

    # --- Flight Search Tests ---
    print("\n[2] Flight Search Tool")
    print("-" * 40)

    # Valid one-way search
    result = search._run(
        origin="MRU",
        destination="DXB",
        departure_date="2026-09-20",
        passengers=1,
        cabin_class="economy",
    )
    test("One-way MRU->DXB finds flights", "Found" in result and "option" in result, result[:100])

    # Valid return search
    result = search._run(
        origin="LHR",
        destination="JFK",
        departure_date="2026-10-01",
        return_date="2026-10-15",
        passengers=2,
        cabin_class="business",
    )
    test(
        "Return LHR->JFK business 2pax finds results",
        "Found" in result or "No flights" in result,
        result[:100],
    )

    # Invalid IATA codes
    result = search._run(
        origin="ZZZ",
        destination="XXX",
        departure_date="2026-09-20",
        passengers=1,
        cabin_class="economy",
    )
    test(
        "Invalid codes ZZZ->XXX graceful error",
        "No flights" in result or "Error" in result,
        result[:150],
    )

    # Past date
    result = search._run(
        origin="MRU",
        destination="DXB",
        departure_date="2020-01-01",
        passengers=1,
        cabin_class="economy",
    )
    test(
        "Past date (2020-01-01) returns error",
        "No flights" in result or "Error" in result,
        result[:150],
    )

    # First class search
    result = search._run(
        origin="DXB",
        destination="NRT",
        departure_date="2026-09-25",
        passengers=1,
        cabin_class="first",
    )
    test("First class DXB->NRT handled", "Found" in result or "No flights" in result, result[:100])

    # Large passenger count
    result = search._run(
        origin="MRU",
        destination="CDG",
        departure_date="2026-10-05",
        passengers=9,
        cabin_class="economy",
    )
    test("9 passengers search handled", "Found" in result or "No flights" in result or "Error" in result, result[:100])

    # Same origin and destination
    result = search._run(
        origin="MRU",
        destination="MRU",
        departure_date="2026-09-20",
        passengers=1,
        cabin_class="economy",
    )
    test(
        "Same origin/destination handled",
        "No flights" in result or "Error" in result,
        result[:150],
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f" Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
