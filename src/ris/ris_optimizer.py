from __future__ import annotations

import numpy as np


class RISOptimizer:
    """Phase alignment and finite-resolution phase quantization for a RIS."""

    def __init__(self, num_elements: int = 64, phase_bits: int = 3) -> None:
        if num_elements < 1 or phase_bits < 1:
            raise ValueError("num_elements and phase_bits must be positive")
        self.num_elements = num_elements
        self.phase_bits = phase_bits

    def quantize_phase(self, coefficients: np.ndarray) -> np.ndarray:
        levels = 2 ** self.phase_bits
        phase = np.mod(np.angle(coefficients), 2 * np.pi)
        indices = np.rint(phase / (2 * np.pi) * levels) % levels
        return np.exp(1j * 2 * np.pi * indices / levels)

    def optimize_phase(self, channel_response: np.ndarray) -> np.ndarray:
        values = np.asarray(channel_response).reshape(-1)
        if values.size == 0:
            return np.ones(self.num_elements, dtype=np.complex128)
        aligned = np.exp(-1j * np.angle(values[: self.num_elements]))
        if aligned.size < self.num_elements:
            aligned = np.pad(aligned, (0, self.num_elements - aligned.size), constant_values=1)
        return self.quantize_phase(aligned)
