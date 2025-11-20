"""
Profile test for THRML-ACT based on profile_act.cpp
Uses EEG-scale parameters and synthetic test signal
"""

import time
from typing import List, Tuple

import jax.numpy as jnp
import numpy as np

from act import (
    ChirpletParams,
    ParameterRanges,
    THRMLChirpletOptimizer,
    generate_chirplet,
    sample_chirplet_parameters,
    bfgs_refine,
)


def print_separator(title: str) -> None:
    """Print a formatted separator line."""
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print('=' * 60)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Calculate SNR in dB."""
    signal_power = np.sum(signal ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power > 0:
        return 10 * np.log10(signal_power / noise_power)
    else:
        return float('inf')


def generate_eeg_test_signal(length: int, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic EEG-like signal with multiple frequency components.
    
    Returns:
        (clean_signal, noisy_signal)
    """
    t = np.arange(length) / fs
    
    # Create a more complex signal with EEG-relevant frequencies
    clean_signal = np.zeros(length, dtype=np.float64)
    
    # Alpha rhythm (8-12 Hz) - dominant
    clean_signal += 2.0 * np.sin(2 * np.pi * 10.0 * t) * np.exp(-0.5 * ((t - 1.0) / 0.5)**2)
    
    # Beta rhythm (12-30 Hz) with chirp
    beta_phase = 2 * np.pi * (15.0 * t + 5.0 * t**2)
    clean_signal += 1.5 * np.cos(beta_phase) * np.exp(-0.5 * ((t - 0.5) / 0.3)**2)
    
    # Theta rhythm (4-8 Hz)
    clean_signal += 1.0 * np.sin(2 * np.pi * 6.0 * t)
    
    # Add noise
    rng = np.random.default_rng(42)
    noise = 0.3 * rng.normal(size=length)
    noisy_signal = clean_signal + noise
    
    return clean_signal, noisy_signal


def run_profile_test() -> None:
    """Run profiling test similar to profile_act.cpp."""
    print_separator("PYTHON THRML-ACT PROFILING TEST - EEG SCALE")
    
    # EEG-typical parameters (matching profile_act.cpp)
    FS = 256.0  # 256 Hz sampling rate
    SIGNAL_LENGTH = 512  # 2 seconds of data
    TRANSFORM_ORDER = 10  # Higher order for detailed analysis
    
    print(f"Configuration:")
    print(f"  Sampling Rate: {FS} Hz")
    print(f"  Signal Length: {SIGNAL_LENGTH} samples ({SIGNAL_LENGTH/FS:.1f} seconds)")
    print(f"  Transform Order: {TRANSFORM_ORDER} chirplets")
    
    # Define EEG-appropriate parameter ranges (matching C++)
    # Note: C++ uses step sizes, we'll convert to min/max for THRML-ACT
    eeg_ranges = ParameterRanges(
        tc_min=0,
        tc_max=SIGNAL_LENGTH - 1,
        fc_min=0.5,
        fc_max=50.0,  # Covers full EEG spectrum
        logDt_min=-4.0,  # Wider range than default
        logDt_max=-1.0,
        c_min=-20.0,
        c_max=20.0,
    )
    
    # Calculate approximate dictionary size for comparison
    # (THRML-ACT uses 32 levels per parameter by default)
    n_levels = 32
    expected_dict_size = n_levels ** 4
    print(f"  Approximate Dictionary Size: {expected_dict_size} configurations")
    print(f"  (THRML-ACT samples from this space rather than generating all)")
    
    # Generate test signal
    clean_signal, signal = generate_eeg_test_signal(SIGNAL_LENGTH, FS)
    signal_jax = jnp.array(signal, dtype=jnp.float32)
    
    # Calculate input SNR
    input_noise = signal - clean_signal
    input_snr = calculate_snr(clean_signal, input_noise)
    print(f"\nInput SNR: {input_snr:.2f} dB")
    
    print_separator("THRML-ACT DECOMPOSITION")
    
    # Create optimizer with custom ranges
    optimizer = THRMLChirpletOptimizer(signal_jax, sampling_rate=FS)
    optimizer.param_ranges = eeg_ranges  # Override default ranges
    
    # Time the full decomposition
    start_time = time.time()
    
    results = []
    residual = signal_jax.copy()
    
    for comp_idx in range(TRANSFORM_ORDER):
        print(f"\nComponent {comp_idx + 1}/{TRANSFORM_ORDER}:")
        
        # Stage 1: THRML sampling
        stage1_start = time.time()
        sampled_params = sample_chirplet_parameters(
            residual,
            eeg_ranges,
            FS,
            n_levels=32,
            beta=1.0,  # Higher beta for more focused search
            n_samples=2000,  # Fewer samples for speed
            n_warmup=300,
            seed=42 + comp_idx,
        )
        stage1_time = time.time() - stage1_start
        
        # Evaluate samples
        t_vec = optimizer.t
        correlations = []
        for params in sampled_params:
            chirplet = generate_chirplet(
                jnp.array([params.tc, params.fc, params.logDt, params.c]),
                t_vec,
                FS,
            )
            corr = float(jnp.abs(jnp.dot(residual, chirplet)))
            correlations.append(corr)
        
        # Select top candidates
        n_candidates = 5  # Fewer candidates for speed
        top_indices = np.argsort(correlations)[-n_candidates:]
        candidates = [sampled_params[i] for i in top_indices]
        
        print(f"  THRML sampling: {stage1_time:.2f}s, best correlation: {max(correlations):.4f}")
        
        # Stage 2: BFGS refinement
        stage2_start = time.time()
        refined_results = []
        for params in candidates:
            refined_params, corr = bfgs_refine(
                params, residual, t_vec, FS, max_iterations=20
            )
            refined_results.append((refined_params, corr))
        
        best_params, best_corr = max(refined_results, key=lambda x: x[1])
        stage2_time = time.time() - stage2_start
        
        print(f"  BFGS refinement: {stage2_time:.2f}s, refined correlation: {best_corr:.4f}")
        
        # Extract component
        best_chirplet = generate_chirplet(
            jnp.array([best_params.tc, best_params.fc, best_params.logDt, best_params.c]),
            t_vec,
            FS,
        )
        coefficient = jnp.dot(residual, best_chirplet)
        component = coefficient * best_chirplet
        residual = residual - component
        
        results.append((best_params, float(coefficient)))
        
        # Check residual energy
        residual_energy = float(jnp.linalg.norm(residual))
        signal_energy = float(jnp.linalg.norm(signal_jax))
        residual_percent = 100 * residual_energy / signal_energy
        
        print(f"  Residual: {residual_percent:.1f}% of signal energy")
        
        if residual_percent < 10.0:  # Early stopping
            print(f"\nEarly stopping: residual below 10% threshold")
            break
    
    total_time = time.time() - start_time
    
    print_separator("RESULTS SUMMARY")
    
    print(f"\nTotal decomposition time: {total_time:.2f} seconds")
    print(f"Average time per component: {total_time/len(results):.2f} seconds")
    
    print(f"\nExtracted {len(results)} components:")
    for i, (params, coeff) in enumerate(results[:5]):  # Show first 5
        print(f"  {i+1}: tc={params.tc:.1f}, fc={params.fc:.2f} Hz, "
              f"logDt={params.logDt:.2f}, c={params.c:.2f} Hz/s, |a|={abs(coeff):.3f}")
    
    if len(results) > 5:
        print(f"  ... and {len(results) - 5} more components")
    
    # Build reconstruction
    reconstruction = np.zeros(SIGNAL_LENGTH)
    for params, coeff in results:
        chirplet = generate_chirplet(
            jnp.array([params.tc, params.fc, params.logDt, params.c]),
            optimizer.t,
            FS,
        )
        reconstruction += coeff * np.array(chirplet)
    
    # Calculate output SNR
    output_noise = signal - reconstruction
    output_snr = calculate_snr(reconstruction, output_noise)
    
    # Final reconstruction quality
    final_residual_percent = 100 * residual_energy / signal_energy
    print(f"\nSignal quality:")
    print(f"  Input SNR: {input_snr:.2f} dB")
    print(f"  Output SNR: {output_snr:.2f} dB")
    print(f"  SNR improvement: {output_snr - input_snr:.2f} dB")
    print(f"\nFinal residual: {final_residual_percent:.1f}% of signal energy")
    print(f"Reconstruction captures: {100 - final_residual_percent:.1f}% of signal energy")
    
    print_separator("PARAMETER DISTRIBUTION")
    
    # Analyze parameter distribution
    tc_vals = [p.tc for p, _ in results]
    fc_vals = [p.fc for p, _ in results]
    logdt_vals = [p.logDt for p, _ in results]
    c_vals = [p.c for p, _ in results]
    
    print(f"\nParameter ranges found:")
    print(f"  tc: [{min(tc_vals):.1f}, {max(tc_vals):.1f}] samples")
    print(f"  fc: [{min(fc_vals):.2f}, {max(fc_vals):.2f}] Hz")
    print(f"  logDt: [{min(logdt_vals):.2f}, {max(logdt_vals):.2f}]")
    print(f"  c: [{min(c_vals):.2f}, {max(c_vals):.2f}] Hz/s")


def main() -> None:
    run_profile_test()


if __name__ == "__main__":
    main()
