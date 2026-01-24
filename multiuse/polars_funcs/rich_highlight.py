import re

from rich.console import Console
from rich.text import Text


def split_text(text: str) -> list[str]:
    # Try blank lines first
    if "\n\n" in text:
        return [p.strip() for p in text.split("\n\n") if p.strip()]

    # Fall back to sentence splitting
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []

    for s in sentences:
        current.append(s)
        joined = " ".join(current)
        if len(current) >= 4 or len(joined) >= 100:
            chunks.append(joined)
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


def highlight_proper_nouns(text: str) -> Text:
    rich_text = Text()

    # Process line by line to preserve paragraph structure
    for line in text.split("\n"):
        if not line.strip():
            rich_text.append("\n")
            continue

        # Track if we're at sentence start
        sentence_start = True

        words = line.split()
        for i, word in enumerate(words):
            # Check if capitalized and not at sentence start
            if word and word[0].isupper() and not sentence_start:
                rich_text.append(word, style="bold yellow")
            else:
                rich_text.append(word)

            # Add space between words
            if i < len(words) - 1:
                rich_text.append(" ")

            # Update sentence_start if word ends with sentence punctuation
            if word and re.search(r"[.!?]$", word) and not re.match(r"^[A-Z][a-z]?\.$", word):
                sentence_start = True
            else:
                sentence_start = False

        rich_text.append("\n")

    return rich_text


class DisplayOutput:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.console = Console()

    def hightlight(self, text: str, width: int = 100) -> None:
        self.console.print(highlight_proper_nouns("\n\n".join(split_text(text))), width=width)


# Example:
# console = Console()
# console.print(highlight_proper_nouns("\n\n".join(split_text(text_to_split))), width=100)
