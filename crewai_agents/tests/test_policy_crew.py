"""Tests for the PolicyCrew — policy retrieval and visa eligibility."""
import sys
import os

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from crewai_agents.policy_crew import run_policy_crew


def test_policy_question():
    print("=" * 60)
    print(" Test 1: Policy Question (baggage)")
    print("=" * 60)
    result = run_policy_crew("What is the carry-on baggage weight limit?")
    print_result(result)
    return result


def test_visa_question():
    print("\n" + "=" * 60)
    print(" Test 2: Visa Eligibility (Mauritius -> Japan)")
    print("=" * 60)
    result = run_policy_crew(
        "Do I need a visa to travel to Japan?",
        passport_country="Mauritius",
        destination_country="Japan",
    )
    print_result(result)
    return result


def test_visa_required():
    print("\n" + "=" * 60)
    print(" Test 3: Visa Required (India -> France)")
    print("=" * 60)
    result = run_policy_crew(
        "Do I need a visa for France?",
        passport_country="India",
        destination_country="France",
    )
    print_result(result)
    return result


def print_result(result):
    print("\n" + "-" * 40)
    print(f"  Answered: {result['answered']}")
    print(f"  Type: {result['question_type']}")
    print(f"  Sources: {result['sources']}")
    print(f"  Answer: {result['answer'][:300]}...")
    print("-" * 40)


if __name__ == "__main__":
    r1 = test_policy_question()
    r2 = test_visa_question()
    r3 = test_visa_required()

    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (policy/baggage):       {'PASS' if r1['answered'] else 'FAIL'}")
    print(f"  Test 2 (visa-free MRU->JPN):   {'PASS' if r2['answered'] else 'FAIL'}")
    print(f"  Test 3 (visa-req IND->FRA):    {'PASS' if r3['answered'] else 'FAIL'}")
