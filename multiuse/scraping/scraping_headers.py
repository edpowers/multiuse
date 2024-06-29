"""Scraping headers."""


class ScrapingHeaders:
    """Utils for scraping headers."""

    @classmethod
    def return_raw_header_dict(cls, header_str: str = "") -> dict:
        """Return the raw header dict."""
        instance = cls()

        header_str = header_str or instance._return_raw_header_str()

        return instance._parse_raw_header_str(header_str)

    def _return_raw_header_str(self) -> str:
        return """
        User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:127.0) Gecko/20100101 Firefox/127.0
        Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8
        Accept-Language: en-US,en;q=0.5
        Accept-Encoding: gzip, deflate, br, zstd
        DNT: 1
        Sec-GPC: 1
        Connection: keep-alive
        Upgrade-Insecure-Requests: 1
        Sec-Fetch-Dest: document
        Sec-Fetch-Mode: navigate
        Sec-Fetch-Site: same-origin
        Sec-Fetch-User: ?1
        Priority: u=1
        Pragma: no-cache
        Cache-Control: no-cache
        """

    def _parse_raw_header_str(self, raw_header_str: str) -> dict:
        """Parse the raw header string."""
        header_dict = {}
        for line in raw_header_str.split("\n"):
            if line := line.strip():
                try:
                    key, value = line.split(": ", maxsplit=1)
                    header_dict[key] = value
                except ValueError as ve:
                    raise ValueError(f"Error parsing line: {line}") from ve
        return header_dict
