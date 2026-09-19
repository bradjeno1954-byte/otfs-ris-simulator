from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.simulation.ris_ber_analysis import simulate_ris_ber_curve


def main() -> None:
    snr_values = [0, 5, 10, 15, 20, 25]
    results = simulate_ris_ber_curve(snr_db_values=snr_values, num_trials=30)

    output_path = Path("plots/ber_vs_snr_v3_ris.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    snr = [item["snr_db"] for item in results]
    without_ris = [item["ber_without_ris"] for item in results]
    with_ris = [item["ber_with_ris"] for item in results]

    plt.figure(figsize=(8, 5))
    plt.semilogy(snr, without_ris, marker="o", linewidth=2, label="Without RIS")
    plt.semilogy(snr, with_ris, marker="s", linewidth=2, label="With optimized RIS")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.xlabel("SNR [dB]")
    plt.ylabel("BER")
    plt.title("OTFS-RIS BER comparison (v3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")

    for item in results:
        print(
            f"SNR={item['snr_db']:.0f} dB -> "
            f"without RIS: {item['ber_without_ris']:.6e} | "
            f"with RIS: {item['ber_with_ris']:.6e}"
        )


if __name__ == "__main__":
    main()
