from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation.ber_analysis import simulate_ber_curve


def main() -> None:
    snr_values = [0, 5, 10, 15, 20, 25]
    results = simulate_ber_curve(snr_db_values=snr_values, num_trials=30)

    snr = [item["snr_db"] for item in results]
    ber = [item["ber"] for item in results]

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr, ber, marker="o", linewidth=2)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("OTFS-RIS BER vs SNR (v2)")
    plt.tight_layout()
    plt.savefig("plots/ber_vs_snr_v2.png", dpi=200)
    print("Saved plot to plots/ber_vs_snr_v2.png")

    for item in results:
        print(f"SNR={item['snr_db']} dB -> BER={item['ber']:.6e}")


if __name__ == "__main__":
    main()
