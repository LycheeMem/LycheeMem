"""Small date normalization helpers.

The memory pipeline receives dates from datasets, users, and LLM JSON. Keep the
normalization local and dependency-free so retrieval code can compare dates
without relying on one exact surface format.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any


_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def normalize_date_key(value: Any) -> str:
    """Return a validated YYYY-MM-DD date extracted from a loose date string."""
    text = str(value or "").strip()
    if not text:
        return ""

    year_first = re.search(
        r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*(?:日)?(?!\d)",
        text,
    )
    if year_first:
        return _validated_date(*year_first.groups())

    month_day_year = re.search(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if month_day_year:
        month_text, day, year = month_day_year.groups()
        month = _MONTHS.get(month_text.casefold())
        if month:
            return _validated_date(year, str(month), day)

    day_month_year = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?[,]?\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    )
    if day_month_year:
        day, month_text, year = day_month_year.groups()
        month = _MONTHS.get(month_text.casefold())
        if month:
            return _validated_date(year, str(month), day)

    return ""


def extract_date_keys(value: Any) -> list[str]:
    """Extract distinct YYYY-MM-DD dates from free text."""
    text = str(value or "")
    if not text.strip():
        return []

    candidates: list[str] = []
    for match in re.finditer(
        r"(?<!\d)(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})\s*(?:日)?(?!\d)",
        text,
    ):
        candidates.append(_validated_date(*match.groups()))

    for match in re.finditer(
        r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?[,]?\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    ):
        month_text, day, year = match.groups()
        month = _MONTHS.get(month_text.casefold())
        if month:
            candidates.append(_validated_date(year, str(month), day))

    for match in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?[,]?\s+(\d{4})\b",
        text,
        flags=re.IGNORECASE,
    ):
        day, month_text, year = match.groups()
        month = _MONTHS.get(month_text.casefold())
        if month:
            candidates.append(_validated_date(year, str(month), day))

    return [date for date in dict.fromkeys(candidates) if date]


def month_key_from_date(date_key: str) -> str:
    date = normalize_date_key(date_key)
    return date[:7] if date else ""


def _validated_date(year: str, month: str, day: str) -> str:
    try:
        dt = datetime(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return ""
    return dt.date().isoformat()
