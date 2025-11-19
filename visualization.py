import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import signal as scipy_signal


def plot_act_decomposition(
  signal, components, parameters, residual, sampling_rate=256.0, save_path=None
):
  """
  ACT Workbench-style visualization

  Layout:
  - Top: Time-domain signal + components
  - Middle: Spectrograms (signal, reconstruction, residual)
  - Bottom: Time-frequency chirplet paths
  """

  t = np.arange(len(signal)) / sampling_rate
  # TRUE reconstruction (components only, without residual)
  # original_signal = reconstruction + residual
  reconstruction = sum(components) if components else np.zeros_like(signal)

  fig = plt.figure(figsize=(14, 10))
  gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

  # ========== ROW 1: Time Domain ==========
  ax1 = fig.add_subplot(gs[0, :])
  ax1.plot(t, np.real(signal), "k-", alpha=0.6, linewidth=1, label="Signal")
  ax1.plot(t, np.real(reconstruction), "r-", linewidth=1.5, label="Reconstruction")

  # Overlay components with offset
  offset = 0
  for i, comp in enumerate(components):
    offset -= np.max(np.abs(comp)) * 1.2
    ax1.plot(
      t,
      np.real(comp) + offset,
      alpha=0.7,
      linewidth=0.8,
      label=f"C{i + 1}" if i < 5 else None,
    )

  ax1.set_xlabel("Time (s)", fontsize=10)
  ax1.set_ylabel("Amplitude", fontsize=10)
  ax1.set_title("Signal Decomposition (Time Domain)", fontsize=12, fontweight="bold")
  ax1.legend(loc="upper right", fontsize=8, ncol=3)
  ax1.grid(alpha=0.3)

  # ========== ROW 2: Spectrograms ==========
  def compute_spectrogram(x, sr):
    f, t_spec, Sxx = scipy_signal.spectrogram(
      x, sr, nperseg=64, noverlap=56, mode="magnitude"
    )
    return f, t_spec, 10 * np.log10(Sxx + 1e-10)

  # Original signal
  ax2 = fig.add_subplot(gs[1, 0])
  f, t_spec, Sxx = compute_spectrogram(np.real(signal), sampling_rate)
  im2 = ax2.pcolormesh(t_spec, f, Sxx, shading="gouraud", cmap="viridis")
  ax2.set_ylabel("Frequency (Hz)", fontsize=9)
  ax2.set_title("Signal Spectrogram", fontsize=10)
  plt.colorbar(im2, ax=ax2, label="dB")

  # Reconstruction
  ax3 = fig.add_subplot(gs[1, 1])
  f, t_spec, Sxx_recon = compute_spectrogram(np.real(reconstruction), sampling_rate)
  im3 = ax3.pcolormesh(t_spec, f, Sxx_recon, shading="gouraud", cmap="viridis")
  ax3.set_title("Reconstruction", fontsize=10)
  plt.colorbar(im3, ax=ax3, label="dB")

  # Residual
  ax4 = fig.add_subplot(gs[1, 2])
  f, t_spec, Sxx_res = compute_spectrogram(np.real(residual), sampling_rate)
  im4 = ax4.pcolormesh(t_spec, f, Sxx_res, shading="gouraud", cmap="plasma")
  ax4.set_title("Residual", fontsize=10)
  plt.colorbar(im4, ax=ax4, label="dB")

  # ========== ROW 3: Chirplet Paths ==========
  ax5 = fig.add_subplot(gs[2, :])

  # Background: Signal spectrogram
  ax5.pcolormesh(t_spec, f, Sxx, shading="gouraud", cmap="gray", alpha=0.3)

  # Overlay chirplet time-frequency paths
  colors = plt.cm.tab10(np.linspace(0, 1, len(parameters)))

  for i, params in enumerate(parameters):
    tc_sec = params.tc / sampling_rate
    fc_hz = params.fc
    c_rate = params.c
    Dt = np.exp(params.logDt) * 0.1 * (len(signal) / sampling_rate)

    # Chirplet instantaneous frequency: f(t) = fc + c*(t-tc)
    t_chirp = np.linspace(tc_sec - 3 * Dt, tc_sec + 3 * Dt, 100)
    t_chirp = np.clip(t_chirp, 0, t[-1])
    f_inst = fc_hz + c_rate * (t_chirp - tc_sec)

    # Plot path with width indicating duration
    ax5.plot(
      t_chirp,
      f_inst,
      color=colors[i],
      linewidth=2,
      label=f"C{i + 1}: fc={fc_hz:.1f}Hz, c={c_rate:.2f}",
    )

    # Center marker
    ax5.scatter(
      [tc_sec],
      [fc_hz],
      color=colors[i],
      s=100,
      marker="o",
      edgecolor="white",
      linewidth=1.5,
      zorder=10,
    )

  ax5.set_xlabel("Time (s)", fontsize=10)
  ax5.set_ylabel("Frequency (Hz)", fontsize=10)
  ax5.set_title("Chirplet Time-Frequency Paths", fontsize=12, fontweight="bold")
  ax5.legend(loc="upper right", fontsize=7, ncol=2)
  ax5.set_ylim([0, sampling_rate / 2])
  ax5.grid(alpha=0.3)

  # ========== ROW 4: Energy Metrics ==========
  ax6 = fig.add_subplot(gs[3, 0])

  # Component energy distribution
  energies = [np.linalg.norm(c) for c in components]
  total_energy = sum(energies)
  percentages = (
    [100 * e / total_energy for e in energies]
    if total_energy > 0
    else [0] * len(energies)
  )

  ax6.bar(
    range(1, len(energies) + 1), percentages, color=colors[: len(energies)], alpha=0.7
  )
  ax6.set_xlabel("Component", fontsize=9)
  ax6.set_ylabel("Energy (%)", fontsize=9)
  ax6.set_title("Energy Distribution", fontsize=10)
  ax6.grid(axis="y", alpha=0.3)

  # Residual decay
  ax7 = fig.add_subplot(gs[3, 1])

  signal_energy = np.linalg.norm(signal)
  residual_energies = [signal_energy]
  temp_residual = signal.copy()

  for comp in components:
    temp_residual = temp_residual - comp
    residual_energies.append(np.linalg.norm(temp_residual))

  residual_pct = [100 * r / signal_energy for r in residual_energies]

  ax7.plot(range(len(residual_pct)), residual_pct, "o-", linewidth=2, markersize=6)
  ax7.set_xlabel("Components Extracted", fontsize=9)
  ax7.set_ylabel("Residual Energy (%)", fontsize=9)
  ax7.set_title("Reconstruction Convergence", fontsize=10)
  ax7.grid(alpha=0.3)
  ax7.set_ylim([0, 105])

  # SNR metrics
  ax8 = fig.add_subplot(gs[3, 2])

  reconstruction_error = np.linalg.norm(signal - reconstruction)
  snr_db = 20 * np.log10(signal_energy / (reconstruction_error + 1e-10))

  metrics = {
    "SNR (dB)": snr_db,
    "Error (%)": 100 * reconstruction_error / signal_energy,
    "Components": len(components),
  }

  ax8.axis("off")
  metrics_text = "\n".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
  ax8.text(
    0.1,
    0.5,
    metrics_text,
    fontsize=12,
    verticalalignment="center",
    family="monospace",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
  )
  ax8.set_title("Quality Metrics", fontsize=10)

  plt.suptitle(
    "ACT Chirplet Decomposition Analysis", fontsize=14, fontweight="bold", y=0.995
  )

  if save_path:
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

  plt.show()
