from __future__ import annotations

import numpy as np

from src.channel.channel_model import ChannelModel
from src.modulation.otfs import OTFSModulator
from src.receiver.mmse import MMSEReceiver

QPSK_CONSTELLATION = np.array([1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j], dtype=np.complex128) / np.sqrt(2)


def qpsk_distance_decision(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex128)
    distances = np.abs(symbols[..., None] - QPSK_CONSTELLATION[None, ...])
    indices = np.argmin(distances, axis=-1)
    return QPSK_CONSTELLATION[indices]


def simulate_ber(
    snr_db: float,
    num_trials: int = 100,
    num_subcarriers: int = 64,
    num_symbols: int = 16,
    num_paths: int = 4,
    noise_seed: int | None = None,
) -> float:
    """Estimate bit error rate for one SNR value."""
    modem = OTFSModulator(num_subcarriers=num_subcarriers, num_symbols=num_symbols)
    channel = ChannelModel(num_paths=num_paths, seed=42)
    receiver = MMSEReceiver()
    rng = np.random.default_rng(noise_seed)

    total_errors = 0
    total_bits = 0

    for _ in range(num_trials):
        tx = modem.generate_qpsk_symbols(batch_size=1)[0]
        response = channel.generate_response(num_subcarriers, num_symbols)

        noise_power = 10 ** (-snr_db / 10.0)
        noise = np.sqrt(noise_power / 2.0) * (
            rng.normal(size=tx.shape) + 1j * rng.normal(size=tx.shape)
        )

        modulated = modem.modulate(symbols=tx[None, ...])[0]
        received = channel.apply(modulated, response) + noise
        equalized = receiver.detect(received, response, noise_power)
        detected = qpsk_distance_decision(equalized)

        expected_bits = np.unpackbits(np.asarray(tx.real > 0, dtype=np.uint8).view(np.uint8))
        actual_bits = np.unpackbits(np.asarray(detected.real > 0, dtype=np.uint8).view(np.uint8))
        total_errors += int(np.count_nonzero(expected_bits != actual_bits))
        total_bits += expected_bits.size

    return total_errors / max(total_bits, 1)


def simulate_ber_curve(
    snr_db_values: list[float] | tuple[float, ...] = (0, 5, 10, 15, 20),
    num_trials: int = 100,
    num_subcarriers: int = 64,
    num_symbols: int = 16,
    num_paths: int = 4,
) -> list[dict[str, float]]:
    """Run a BER sweep over SNR values and return structured results."""
    results = []
    for snr_db in snr_db_values:
        ber = simulate_ber(
            snr_db=snr_db,
            num_trials=num_trials,
            num_subcarriers=num_subcarriers,
            num_symbols=num_symbols,
            num_paths=num_paths,
        )
        results.append({"snr_db": float(snr_db), "ber": float(ber)})
    return results
