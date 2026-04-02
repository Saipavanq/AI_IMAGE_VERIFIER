"""Model definitions for AI image verifier."""


SUPPORTED_MODELS = ("cnn", "efficientnet", "vit")


def list_models() -> tuple[str, ...]:
    """Return supported model names."""
    return SUPPORTED_MODELS
