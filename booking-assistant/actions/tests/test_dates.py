from datetime import date, timedelta

from actions.actions import normalise_date

DEPARTURE = date(2026, 8, 30)


def test_accepts_iso_input():
    assert normalise_date("2027-03-30", DEPARTURE) == "2027-03-30"


def test_accepts_natural_language_with_ordinal():
    assert normalise_date("30th March 2027", DEPARTURE) == "2027-03-30"


def test_rejects_nonexistent_date():
    assert normalise_date("29th February 2027") is None


def test_rejects_past_date():
    assert normalise_date("01 January 2020") is None


def test_rejects_return_before_departure():
    assert normalise_date("2026-08-20", DEPARTURE) is None


def test_rejects_beyond_booking_horizon():
    assert normalise_date("2028-06-01") is None


def test_accepts_relative_date():
    result = normalise_date("in 10 days")
    assert result == (date.today() + timedelta(days=10)).isoformat()


def test_empty_input_is_none():
    assert normalise_date("") is None
    assert normalise_date(None) is None