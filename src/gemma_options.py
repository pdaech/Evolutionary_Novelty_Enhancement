"""Dependency-free Gemma evaluator configuration shared by CLI and runtime."""

DEFAULT_IMAGE_TOKEN_BUDGET = 280
SUPPORTED_IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def validate_image_token_budget(value: int) -> int:
    """Require one of Gemma 4's supported visual soft-token budgets."""
    if value not in SUPPORTED_IMAGE_TOKEN_BUDGETS:
        supported = ", ".join(str(item) for item in SUPPORTED_IMAGE_TOKEN_BUDGETS)
        raise ValueError(f"Gemma image token budget must be one of: {supported}")
    return value
