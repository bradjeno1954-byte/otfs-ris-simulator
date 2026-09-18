"""Common numerical helpers."""

import numpy as np


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (value_db / 10.0)


def linear_to_db(value_linear: float) -> float:
    return float("-inf") if value_linear <= 0 else 10.0 * np.log10(value_linear)
