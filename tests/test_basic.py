import numpy as np

from src.channel import ChannelModel
from src.modulation import OTFSModulator
from src.ris import RISOptimizer


def test_otfs_round_trip_shape():
    modem = OTFSModulator(8, 4)
    symbols = modem.generate_qpsk_symbols(batch_size=2)
    assert modem.demodulate(modem.modulate(symbols)).shape == symbols.shape


def test_channel_response_shape():
    response = ChannelModel(num_paths=2, seed=1).generate_response(8, 4)
    assert response.shape == (4, 8)


def test_ris_phase_shape_and_unit_modulus():
    phases = RISOptimizer(10, 2).optimize_phase(np.ones((4, 8), dtype=complex))
    assert phases.shape == (10,)
    assert np.allclose(np.abs(phases), 1.0)
