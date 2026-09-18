from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OTFSModulator:
    """A compact delay-Doppler OTFS transform implementation.

    The data grid has shape ``(batch, num_symbols, num_subcarriers)``.
    """

    num_subcarriers: int = 64
    num_symbols: int = 16

    def __post_init__(self) -> None:
        if self.num_subcarriers <= 0 or self.num_symbols <= 0:
            raise ValueError("Grid dimensions must be positive")

    def generate_qpsk_symbols(self, batch_size: int = 1) -> np.ndarray:
        indices = np.random.randint(
            0, 4, size=(batch_size, self.num_symbols, self.num_subcarriers)
        )
        constellation = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j]) / np.sqrt(2)
        return constellation[indices]

    def modulate(self, symbols: np.ndarray | None = None, batch_size: int = 1) -> np.ndarray:
        if symbols is None:
            symbols = self.generate_qpsk_symbols(batch_size)
        symbols = np.asarray(symbols)
        expected = (self.num_symbols, self.num_subcarriers)
        if symbols.ndim != 3 or symbols.shape[-2:] != expected:
            raise ValueError(f"Expected shape (batch, {expected[0]}, {expected[1]})")
        # ISFFT: Doppler dimension uses FFT and delay dimension uses IFFT.
        return np.fft.ifft(np.fft.fft(symbols, axis=-2), axis=-1)

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        signal = np.asarray(signal)
        if signal.ndim != 3 or signal.shape[-2:] != (self.num_symbols, self.num_subcarriers):
            raise ValueError("Signal has an incompatible shape")
        return np.fft.ifft(np.fft.fft(signal, axis=-1), axis=-2)
