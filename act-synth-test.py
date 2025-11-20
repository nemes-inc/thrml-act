import math
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

import act
from act import decompose_signal


# =============================================================================
# Synthetic signal generation (ported from test_act_synthetic.cpp)
# =============================================================================


def chirplet_sample(t: float, tc: float, fc: float, c: float, dt: float, amp: float) -> float:
  """Single Gaussian-enveloped chirp sample (matches C++ chirplet_sample).

  Args:
    t: Time in seconds.
    tc: Time center in seconds.
    fc: Center frequency in Hz.
    c: Chirp rate in Hz/s.
    dt: Duration parameter (seconds).
    amp: Amplitude.
  """
  time_diff = t - tc
  gaussian = math.exp(-0.5 * (time_diff / dt) ** 2)
  phase = 2.0 * math.pi * (c * time_diff * time_diff + fc * time_diff)
  return amp * gaussian * math.cos(phase)


def ground_truth_params(length: int) -> List[List[float]]:
  """Ground truth parameters as in C++ ground_truth_params.

  Returns a list of [tc_samples, fc_hz, logDt, c].
  """
  tc_step_samples = max(1.0, math.floor(length / 64.0))

  def tc_samples(k: int) -> float:
    return k * tc_step_samples

  def fc_val(hz_qtr: float) -> float:
    return 0.25 * round(hz_qtr / 0.25)

  def logdt_val(logdt: float) -> float:
    return logdt

  def c_val(c: float) -> float:
    return round(c)

  gt: List[List[float]] = []
  gt.append([tc_samples(16), fc_val(10.0), logdt_val(-2.0), c_val(8.0)])
  gt.append([tc_samples(48), fc_val(6.0), logdt_val(-1.5), c_val(-6.0)])
  return gt


def generate_synthetic_signal(length: int, fs: float) -> np.ndarray:
  """Generate the same clean synthetic signal as generate_synthetic_signal in C++.

  This uses un-normalized chirplets with fixed amplitudes.
  """
  sig = np.zeros(length, dtype=np.float64)

  tc_step_samples = max(1.0, math.floor(length / 64.0))

  def tc_sec(k: int) -> float:
    return (k * tc_step_samples) / fs

  def fc_val(hz_qtr: float) -> float:
    return 0.25 * round(hz_qtr / 0.25)

  def dt_from_log(logdt: float) -> float:
    return math.exp(logdt)

  def c_val(c: float) -> float:
    return round(c)

  # First chirplet
  tc1 = tc_sec(16)
  fc1 = fc_val(10.0)
  c1 = c_val(8.0)
  dt1 = dt_from_log(-2.0)
  a1 = 0.9

  # Second chirplet
  tc2 = tc_sec(48)
  fc2 = fc_val(6.0)
  c2 = c_val(-6.0)
  dt2 = dt_from_log(-1.5)
  a2 = 0.7

  for i in range(length):
    t = i / fs
    sig[i] += chirplet_sample(t, tc1, fc1, c1, dt1, a1)
    sig[i] += chirplet_sample(t, tc2, fc2, c2, dt2, a2)

  return sig


def compute_snr_db(clean: np.ndarray, approx: np.ndarray) -> float:
  """Compute SNR in dB as in the C++ test (using clean vs approx)."""
  clean_energy = float(np.sum(clean**2))
  clean_resid_energy = float(np.sum((clean - approx) ** 2))
  return 10.0 * math.log10((clean_energy + 1e-12) / (clean_resid_energy + 1e-12))


# =============================================================================
# THRML-ACT synthetic test harness
# =============================================================================


def run_thrml_act_synthetic(fs: float = 128.0, length: int = 256) -> None:
  """Run a synthetic test similar in spirit to test_act_synthetic.cpp.

  This reuses thrml-act's decompose_signal on the same synthetic signal
  and reports input/output SNR and recovered vs ground-truth parameters.
  """
  # 1) Clean synthetic signal
  clean = generate_synthetic_signal(length, fs)

  # 2) Additive white Gaussian noise at 0 dB input SNR (same logic as C++)
  target_input_snr_db = 0.0
  clean_energy = float(np.sum(clean**2))
  clean_power = clean_energy / float(length)
  noise_power = clean_power / (10.0 ** (target_input_snr_db / 10.0))
  noise_std = math.sqrt(noise_power)

  rng = np.random.default_rng(42)
  noise = rng.normal(loc=0.0, scale=noise_std, size=length)

  signal = clean + noise

  noisy_energy = float(np.sum(signal**2))
  input_snr_db = 10.0 * math.log10(
    (clean_energy + 1e-12) / (noisy_energy - clean_energy + 1e-12)
  )

  print(f"Input SNR: {input_snr_db:.2f} dB (target {target_input_snr_db:.1f} dB)")

  # 3) Run thrml-act decomposition (2 components like C++ ground truth)
  signal_jax = jnp.array(signal, dtype=jnp.float32)

  components, params_list, residual = decompose_signal(
    signal_jax,
    sampling_rate=fs,
    n_components=2,
    residual_threshold=0.0,
    use_thrml=True,
  )

  # Reconstruction
  if components:
    recon = jnp.sum(jnp.stack(components, axis=0), axis=0)
  else:
    recon = jnp.zeros_like(signal_jax)

  recon_np = np.asarray(recon, dtype=np.float64)

  # 4) Output SNR and improvement vs input
  output_snr_db = compute_snr_db(clean, recon_np)
  improvement_db = output_snr_db - input_snr_db

  print(f"Output SNR (THRML+ACT): {output_snr_db:.2f} dB, Improvement: {improvement_db:.2f} dB")

  # 5) Recovered vs ground truth parameter comparison (best-effort matching)
  gt = ground_truth_params(length)

  # Convert recovered params to plain lists [tc_samples, fc_hz, logDt, c]
  rec_params: List[List[float]] = [
    [float(p.tc), float(p.fc), float(p.logDt), float(p.c)] for p in params_list
  ]

  print("Recovered vs Truth (THRML+ACT, up to first 2 components):")
  kmax = min(2, min(len(rec_params), len(gt)))
  gt_used = [False] * len(gt)

  for r in range(kmax):
    rp = rec_params[r]
    best_idx = -1
    best_dist = 1e300
    for gti, g in enumerate(gt):
      if gt_used[gti]:
        continue
      dtc = rp[0] - g[0]
      dfc = rp[1] - g[1]
      dld = rp[2] - g[2]
      dc = rp[3] - g[3]
      dist = dtc * dtc + dfc * dfc + dld * dld + dc * dc
      if dist < best_dist:
        best_dist = dist
        best_idx = gti

    if best_idx >= 0:
      gt_used[best_idx] = True
      g = gt[best_idx]
      dtc = rp[0] - g[0]
      dfc = rp[1] - g[1]
      dld = rp[2] - g[2]
      dc = rp[3] - g[3]
      print(
        f"  Atom {r+1}: rec(tc={rp[0]:.2f}, fc={rp[1]:.2f}, logDt={rp[2]:.3f}, c={rp[3]:.2f}) | "
        f"gt(tc={g[0]:.2f}, fc={g[1]:.2f}, logDt={g[2]:.3f}, c={g[3]:.2f}) | "
        f"d(tc,fc,logDt,c)=({dtc:.2f}, {dfc:.2f}, {dld:.3f}, {dc:.2f})"
      )


def main() -> None:
  run_thrml_act_synthetic()


if __name__ == "__main__":
  main()
