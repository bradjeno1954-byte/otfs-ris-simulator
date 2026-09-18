# OTFS-RIS Simulator

End-to-end PyTorch simulator for OTFS modulation with Reconfigurable Intelligent Surface (RIS) optimization.

## Features

- Basic OTFS delay-Doppler modulation/demodulation
- Stochastic multipath channel model
- Quantized RIS phase optimization
- Reproducible examples and tests

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run the example

```bash
python examples/basic_simulation.py
```

## Run tests

```bash
pytest
```

## Structure

```text
config/       Configuration files
src/          Simulator source code
examples/     Runnable examples
tests/        Automated tests
docs/         Documentation
```

## Status

This is the initial functional skeleton. Future versions will add a complete waveform-level OTFS chain, BER/SNR sweeps, MMSE detection, and trainable RIS optimization.

## License

MIT
