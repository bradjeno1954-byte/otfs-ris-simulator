"""Basic end-to-end example."""

from src.channel import ChannelModel
from src.modulation import OTFSModulator
from src.ris import RISOptimizer


def main() -> None:
    modulator = OTFSModulator(num_subcarriers=64, num_symbols=16)
    channel = ChannelModel(num_paths=4, seed=42)
    ris = RISOptimizer(num_elements=64, phase_bits=3)

    transmitted = modulator.modulate(batch_size=1)
    response = channel.generate_response(64, 16)
    received = channel.apply(transmitted, response)
    phases = ris.optimize_phase(response)

    print(f"transmitted: {transmitted.shape}")
    print(f"received:    {received.shape}")
    print(f"RIS phases:  {phases.shape}")


if __name__ == "__main__":
    main()
