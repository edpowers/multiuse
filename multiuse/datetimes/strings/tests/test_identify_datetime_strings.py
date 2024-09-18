import unittest

from multiuse.datetimes.strings.identify_datetime_strings import IdentifyDatetimeStrings


class TestIdentifyDatetimeStrings(unittest.TestCase):
    def test_find_dates(self):
        test_cases = [
            ("3 months ending Feb. 28, 2015", 2),
            ("Sep. 30, 2015", 1),
            ("9 Months Ended", 1),
            ("9", 0),
            ("Nine", 0),
            ("'Three Months Ended September 30, 2014'", 2),
            ("September 30, 2015", 1),
        ]
        for input_string, expected_count in test_cases:
            with self.subTest(input_string=input_string):
                self.assertEqual(
                    len(IdentifyDatetimeStrings.find_dates(input_string)),
                    expected_count,
                )

    def test_find_time_periods(self):
        self.assertEqual(
            len(
                IdentifyDatetimeStrings.find_time_periods(
                    "3 months ending Feb. 28, 2015"
                )
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
