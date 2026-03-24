from collections import defaultdict
from collections.abc import Sequence


def normalize_to_base_terms(
    terms: Sequence[str],
    minimum_word_length_match: int = 3,
) -> list[str]:
    """Collapse similar long terms to their 4-word base."""

    base_groups: defaultdict[str, list[str]] = defaultdict(list)
    short_terms = []

    for term in terms:
        words = term.split()

        if len(words) <= minimum_word_length_match:
            short_terms.append(term)
            continue

        # Extract first 4 words as base
        base = " ".join(words[:minimum_word_length_match])
        base_groups[base].append(term)

    # Keep only base if multiple variants exist
    result = short_terms.copy()
    for base, variants in base_groups.items():
        if len(variants) > 1:
            result.append(base)  # Just the base
        else:
            result.append(variants[0])  # Keep full term if singleton

    return sorted(set(result))


def filter_redundant_long_terms(
    terms: Sequence[str],
    minimum_word_length_match: int = 3,
) -> list[str]:
    """Remove lengthy terms that contain shorter complete phrases."""

    terms_set = set(terms)
    filtered = []

    for term in sorted(terms, key=len):  # Process shortest first
        if not isinstance(term, str):
            ve_msg = f"Term is not string. Found {type(term)=}"
            raise ValueError(ve_msg)

        words = term.split()

        # Keep all terms ≤4 words
        if len(words) <= minimum_word_length_match:
            filtered.append(term)
            continue

        # Check if any 4-word substring exists as standalone term
        is_redundant = False
        for i in range(len(words) - (minimum_word_length_match - 1)):
            four_word = " ".join(words[i : i + minimum_word_length_match])
            if four_word in terms_set and four_word != term:
                is_redundant = True
                break

        if not is_redundant:
            filtered.append(term)

    return filtered
