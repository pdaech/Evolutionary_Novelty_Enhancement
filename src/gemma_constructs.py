"""Versioned, image-only wording for the six Gemma fitness conditions."""

ADJECTIVES = {
    "novelty": "novel",
    "unusualness": "unusual",
    "uncommonness": "uncommon",
    "uniqueness": "unique",
    "originality": "original",
    "innovation": "innovative",
}


def scoring_prompt(construct: str) -> str:
    """Change only the adjective; keep scale, decoding target and context fixed."""
    try:
        adjective = ADJECTIVES[construct]
    except KeyError as exc:
        raise ValueError(f"Unknown Gemma fitness construct: {construct!r}") from exc
    return (
        f"How {adjective} do you find the image? "
        "Use a continuous score from 1 to 5, where higher scores mean that you "
        f"find the image more {adjective}. Return only valid JSON in this exact "
        'shape: {"score": <number>}'
    )
