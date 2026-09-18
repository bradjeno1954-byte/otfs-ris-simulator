from __future__ import annotations

import numpy as np


class ChannelModel:
    """Sparse complex multipath channel on a delay-Doppler grid."""

    def __init__(self, num_paths: int = 4, seed: int | None = None) -> None:
        if num_paths < 1:
            raise ValueError("num_paths must be at least one")
        self.num_paths = num_paths
        self.rng = np.random.default_rng(seed)

    def generate_response(self, num_subcarriers: int = 64, num_symbols: int = 16) -> np.ndarray:
        gains = self.rng.normal(size=self.num_paths) + 1j * self.rng.normal(size=self.num_paths)
        gains /= np.linalg.norm(gains)
        response = np.zeros((num_symbols, num_subcarriers), dtype=np.complex128)
        delays = self.rng.integers(0, num_subcarriers, self.num_paths)
        dopplers = self.rng.integers(0, num_symbols, self.num_paths)
        for gain, delay, doppler in zip(gains, delays, dopplers):
            response[doppler, delay] += gain
        return response

    @staticmethod
    def apply(signal: np.ndarray, response: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal)
        response = np.asarray(response)
        if signal.shape[-2:] != response.shape:
            raise ValueError("Signal and response grid shapes must match")
        return signal * response
