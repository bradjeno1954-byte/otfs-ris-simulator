from __future__ import annotations

from src.simulation.ber_analysis import simulate_ber_curve


def main() -> None:
    snr_values = [0, 5, 10, 15, 20]
    results = simulate_ber_curve(snr_db_values=snr_values, num_trials=20)
    for item in results:
        print(f"SNR={item['snr_db']} dB -> BER={item['ber']:.6e}")


if __name__ == "__main__":
    main()
