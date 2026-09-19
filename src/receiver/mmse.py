from __future__ import annotations

import numpy as np


class MMSEReceiver:
    """Element-wise MMSE equalizer for the current grid channel model."""

    def __init__(self, eps: float = 1e-12) -> None:
        self.eps = eps

    def detect(
        self,
        received: np.ndarray,
        channel_response: np.ndarray,
        noise_var: float,
    ) -> np.ndarray:
        """Equalize a received complex grid using an MMSE coefficient."""
        received = np.asarray(received, dtype=np.complex128)
        channel_response = np.asarray(channel_response, dtype=np.complex128)

        if received.shape != channel_response.shape:
            raise ValueError("received and channel_response must have identical shapes")
        if noise_var < 0:
            raise ValueError("noise_var must be non-negative")

        equalizer = np.conj(channel_response) / (
            np.abs(channel_response) ** 2 + noise_var + self.eps
        )
        return equalizer * received

    @staticmethod
    def qpsk_decision(symbols: np.ndarray) -> np.ndarray:
        """Map complex estimates to the nearest normalized QPSK symbol."""
        symbols = np.asarray(symbols, dtype=np.complex128)
        constellation = np.array(
            [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex128
        ) / np.sqrt(2)
        distances = np.abs(symbols[..., None] - constellation)
        return constellation[np.argmin(distances, axis=-1)]
