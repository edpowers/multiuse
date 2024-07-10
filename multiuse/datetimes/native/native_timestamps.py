"""Implementation for native timestamps within Python."""

import datetime


class NativeTimestamps:
    @staticmethod
    def get_current_timestamp_iso() -> str:
        """
        Returns the current timestamp in ISO 8601 format.

        Returns:
            str: The current timestamp in ISO 8601 format.
        """
        current_timestamp = datetime.datetime.now()
        return current_timestamp.isoformat()

    @staticmethod
    def get_current_timestamp_unix() -> int:
        """
        Returns the current unix timestamp info.
        """
        current_timestamp = datetime.datetime.now()
        return int(current_timestamp.timestamp())
