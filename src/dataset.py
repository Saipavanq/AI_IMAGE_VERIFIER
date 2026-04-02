"""Dataset utilities for AI image verifier."""


def get_dataset_info() -> dict:
    """Return basic dataset folder expectations."""
    return {
        "train_real": "data/raw/train/REAL",
        "train_fake": "data/raw/train/FAKE",
        "test_real": "data/raw/test/REAL",
        "test_fake": "data/raw/test/FAKE",
    }
