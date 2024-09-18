"""Identifying datetime strings."""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from dateutil.parser import parse


class IdentifyDatetimeStrings:
    DATE_PATTERNS = [
        # r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+?\d{4}?\b",
        (
            r"\b(?:January|February|March|April|May|June|July|August|)"
            r"September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+?\d{4}?\b"
        ),
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b",
        (
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|"
            r"May|June|July|August|September|October|November|December)\s+\d{4}\b"
        ),
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{4}\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
    ]

    TIME_PERIOD_PATTERNS = [
        r"\b(?:[369]|three|six|nine)\s+months\s+end[inged]*\b",
        r"\bQ[1-4]\b",
        r"\b(?:first|second|third|fourth)\s+half\s+of\s+the\s+year\b",
    ]

    @classmethod
    def find_dates(cls, text: str) -> List[str]:
        combined_pattern = "|".join(cls.DATE_PATTERNS)
        return re.findall(combined_pattern, text, re.IGNORECASE)

    @classmethod
    def find_time_periods(cls, text: str) -> List[str]:
        combined_pattern = "|".join(cls.TIME_PERIOD_PATTERNS)
        return re.findall(combined_pattern, text, re.IGNORECASE)

    @classmethod
    def parse_date(cls, date_string: str) -> Optional[datetime]:
        try:
            return parse(date_string, fuzzy=True)
        except ValueError:
            return None

    @classmethod
    def parse_time_period(
        cls, period_string: str
    ) -> Tuple[Optional[int], Optional[str]]:
        period_string = period_string.lower()
        if "month" in period_string:
            if months := re.search(r"\d+|three|six|nine", period_string):
                return (
                    int(months.group())
                    if months.group().isdigit()
                    else {
                        "three": 3,
                        "six": 6,
                        "nine": 9,
                    }[months.group()]
                ), "months"
        elif period_string.startswith("q"):
            return int(period_string[1]), "quarter"
        elif "half" in period_string:
            return 1 if "first" in period_string else 2, "half"
        return None, None

    @classmethod
    def extract_datetime_info(
        cls, text: str
    ) -> Tuple[Optional[datetime], Optional[Tuple[Optional[int], Optional[str]]]]:
        dates = cls.find_dates(text)
        time_periods = cls.find_time_periods(text)

        parsed_date = cls.parse_date(dates[0]) if dates else None
        parsed_period = None
        if time_periods:
            parsed_period = cls.parse_time_period(time_periods[0])

        return parsed_date, parsed_period
