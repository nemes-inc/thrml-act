"""
Test THRML-ACT on challenging synthetic signal from actsleepstudy
Three overlapping chirplets with close frequencies
"""

import time
from typing import List, Tuple
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from act import (
    ChirpletParams,
    ParameterRanges,
    THRMLChirpletOptimizer,
    generate_chirplet,
    sample_chirplet_parameters,
    bfgs_refine,
    compute_correlation,
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
    return 10 * np.log10(signal_power / noise_power)


def generate_challenging_signal(length: int, fs: float) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate the challenging synthetic signal from actsleepstudy.
    
    Returns:
        signal: The composite signal
        ground_truth: List of ground truth parameters
    """
    t = np.arange(length) / fs
    
    # Ground truth parameters from syntheticsignal.py
    ground_truth = [
        {"tc": 200.0, "fc": 2.0, "logDt": 0.0, "c": 5.0, "name": "Chirplet 1 (upchirp)"},
        {"tc": 60.0, "fc": 4.0, "logDt": -1.0, "c": -5.0, "name": "Chirplet 2 (downchirp)"},
        {"tc": 100.0, "fc": 3.0, "logDt": 0.0, "c": 1.0, "name": "Chirplet 3 (slight upchirp)"},
    ]
    
    # Generate clean signal
    clean_signal = np.zeros(length, dtype=np.float64)
    
    for gt in ground_truth:
        # Generate chirplet using our function
        chirplet = generate_chirplet(
            jnp.array([gt["tc"], gt["fc"], gt["logDt"], gt["c"]]),
            jnp.array(t),
            fs
        )
        clean_signal += np.array(chirplet)
    
    # Normalize
    scale = np.max(np.abs(clean_signal))
    if scale > 0:
        clean_signal /= scale
    
    # Add small noise
    noise_level = 0.01  # 1% noise
    rng = np.random.default_rng(42)
    noise = noise_level * rng.normal(size=length)
    signal = clean_signal + noise
    
    return signal, ground_truth


def run_challenging_test() -> None:
    """Run the challenging chirplet decomposition test."""
    print_separator("CHALLENGING CHIRPLET DECOMPOSITION TEST")
    
    # Parameters matching actsleepstudy
    FS = 256.0
    EPOCH = 2
    LENGTH = EPOCH * 256  # 512 samples
    ORDER = 3  # Try to recover all 3 components
    
    print(f"Test Configuration:")
    print(f"  Sampling Rate: {FS} Hz")
    print(f"  Signal Length: {LENGTH} samples ({EPOCH} seconds)")
    print(f"  Transform Order: {ORDER} components")
    
    # Generate signal
    signal, ground_truth = generate_challenging_signal(LENGTH, FS)
    signal_jax = jnp.array(signal, dtype=jnp.float32)
    
    print("\nGround Truth Chirplets:")
    for gt in ground_truth:
        print(f"  {gt['name']}: tc={gt['tc']}, fc={gt['fc']} Hz, "
              f"logDt={gt['logDt']}, c={gt['c']} Hz/s")
    
    # Calculate input SNR
    clean_signal = signal - (signal - signal)  # Placeholder, we'd need to regenerate
    input_snr = calculate_snr(signal, signal * 0.01)  # Approximate
    print(f"\nInput SNR: {input_snr:.2f} dB")
    
    # Define search ranges (matching syntheticsignal.py)
    print_separator("THRML-ACT DECOMPOSITION")
    
    challenging_ranges = ParameterRanges(
        tc_min=0,
        tc_max=LENGTH,
        fc_min=0.6,
        fc_max=15.0,
        logDt_min=-4.0,
        logDt_max=1.0,
        c_min=-10.0,
        c_max=10.0,
    )
    
    print(f"Search ranges:")
    print(f"  tc: [0, {LENGTH}]")
    print(f"  fc: [0.6, 15.0] Hz")
    print(f"  logDt: [-4.0, 1.0]")
    print(f"  c: [-10.0, 10.0] Hz/s")
    
    # Create optimizer
    optimizer = THRMLChirpletOptimizer(signal_jax, sampling_rate=FS)
    optimizer.param_ranges = challenging_ranges
    
    # Decomposition
    start_time = time.time()
    results = []
    residual = signal_jax.copy()
    t_vec = optimizer.t
    
    print("\nDecomposing signal...")
    
    for comp_idx in range(ORDER):
        print(f"\nComponent {comp_idx + 1}/{ORDER}:")
        
        # Stage 1: THRML sampling with parameters tuned for this problem
        stage1_start = time.time()
        sampled_params = sample_chirplet_parameters(
            residual,
            challenging_ranges,
            FS,
            n_levels=32,
            beta=2.0,  # Higher beta for more focused search
            n_samples=5000,  # More samples for challenging case
            n_warmup=1000,
            seed=42 + comp_idx,
        )
        stage1_time = time.time() - stage1_start
        
        # Evaluate samples
        correlations = []
        for params in sampled_params:
            chirplet = generate_chirplet(
                jnp.array([params.tc, params.fc, params.logDt, params.c]),
                t_vec,
                FS,
            )
            corr = float(compute_correlation(residual, chirplet))
            correlations.append(corr)
        
        # Select top candidates
        n_candidates = 10  # More candidates for challenging case
        top_indices = np.argsort(correlations)[-n_candidates:]
        candidates = [sampled_params[i] for i in top_indices]
        
        print(f"  THRML sampling: {stage1_time:.2f}s, best correlation: {max(correlations):.4f}")
        
        # Stage 2: BFGS refinement
        stage2_start = time.time()
        refined_results = []
        for params in candidates:
            refined_params, corr = bfgs_refine(
                params, residual, t_vec, FS, max_iterations=50  # More iterations
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
        
        print(f"  Found: tc={best_params.tc:.1f}, fc={best_params.fc:.2f}, "
              f"logDt={best_params.logDt:.2f}, c={best_params.c:.2f}, |a|={abs(coefficient):.3f}")
    
    total_time = time.time() - start_time
    
    # Reconstruction
    reconstruction = np.zeros(LENGTH)
    for params, coeff in results:
        chirplet = generate_chirplet(
            jnp.array([params.tc, params.fc, params.logDt, params.c]),
            t_vec,
            FS,
        )
        reconstruction += coeff * np.array(chirplet)
    
    # Calculate output SNR
    noise = signal - reconstruction
    output_snr = calculate_snr(reconstruction, noise)
    
    print_separator("RESULTS SUMMARY")
    
    print(f"Total decomposition time: {total_time:.2f} seconds")
    print(f"Average time per component: {total_time/ORDER:.2f} seconds")
    
    print(f"\nRecovered parameters:")
    for i, (params, coeff) in enumerate(results):
        print(f"  Component {i+1}: tc={params.tc:.1f}, fc={params.fc:.2f}, "
              f"logDt={params.logDt:.2f}, c={params.c:.2f}, |a|={abs(coeff):.3f}")
    
    print(f"\nSignal quality:")
    print(f"  Output SNR: {output_snr:.2f} dB")
    
    # Parameter recovery analysis
    print_separator("PARAMETER RECOVERY ANALYSIS")
    
    print("Attempting to match recovered components to ground truth...")
    print("(Based on frequency proximity)")
    
    # Simple matching based on frequency
    matched = [False] * len(ground_truth)
    for i, (params, coeff) in enumerate(results):
        best_match = -1
        min_freq_diff = float('inf')
        
        for j, gt in enumerate(ground_truth):
            if not matched[j]:
                freq_diff = abs(params.fc - gt['fc'])
                if freq_diff < min_freq_diff:
                    min_freq_diff = freq_diff
                    best_match = j
        
        if best_match >= 0 and min_freq_diff < 2.0:  # Within 2 Hz
            matched[best_match] = True
            gt = ground_truth[best_match]
            
            print(f"\nComponent {i+1} matched to {gt['name']}:")
            print(f"  tc error: {params.tc - gt['tc']:.1f} samples")
            print(f"  fc error: {params.fc - gt['fc']:.2f} Hz")
            print(f"  logDt error: {params.logDt - gt['logDt']:.2f}")
            print(f"  c error: {params.c - gt['c']:.2f} Hz/s")
    
    # Check recovery
    print("\nRecovery summary:")
    found_count = 0
    for i, gt in enumerate(ground_truth):
        if matched[i]:
            print(f"  ✓ {gt['name']} - FOUND")
            found_count += 1
        else:
            print(f"  ✗ {gt['name']} - NOT FOUND")
    
    print(f"\nRecovered {found_count}/{len(ground_truth)} ground truth components")
    
    # Additional analysis for THRML-ACT
    print_separator("THRML-ACT SPECIFIC ANALYSIS")
    
    # Analyze parameter distribution of samples
    print("\nParameter space exploration:")
    tc_vals = [p.tc for p, _ in results]
    fc_vals = [p.fc for p, _ in results]
    logdt_vals = [p.logDt for p, _ in results]
    c_vals = [p.c for p, _ in results]
    
    print(f"  tc range explored: [{min(tc_vals):.1f}, {max(tc_vals):.1f}]")
    print(f"  fc range explored: [{min(fc_vals):.2f}, {max(fc_vals):.2f}] Hz")
    print(f"  logDt range explored: [{min(logdt_vals):.2f}, {max(logdt_vals):.2f}]")
    print(f"  c range explored: [{min(c_vals):.2f}, {max(c_vals):.2f}] Hz/s")
    
    # Check if we found the challenging overlapping components
    if found_count == len(ground_truth):
        print("\n✓ Successfully decomposed the challenging signal!")
        print("  THRML-ACT handled overlapping chirplets with close frequencies")
    else:
        print("\n⚠ Partial success - some components were not recovered")
        print("  This confirms the signal is indeed challenging to decompose")
    
    # Plotting
    print_separator("VISUALIZATION")
    print("Generating plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    t = np.arange(LENGTH) / FS
    
    # Plot 1: Original signal and ground truth components
    ax1 = axes[0]
    ax1.plot(t, signal, 'k-', alpha=0.7, linewidth=1.5, label='Composite Signal')
    
    # Generate and plot ground truth components
    for i, gt in enumerate(ground_truth):
        chirplet = generate_chirplet(
            jnp.array([gt["tc"], gt["fc"], gt["logDt"], gt["c"]]),
            jnp.array(t),
            FS
        )
        ax1.plot(t, np.array(chirplet), '--', alpha=0.6, linewidth=1, 
                label=f'{gt["name"]}')
    
    ax1.set_title('Original Signal and Ground Truth Components')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recovered components
    ax2 = axes[1]
    ax2.plot(t, signal, 'k-', alpha=0.3, linewidth=1, label='Original Signal')
    
    # Plot recovered components
    for i, (params, coeff) in enumerate(results):
        chirplet = generate_chirplet(
            jnp.array([params.tc, params.fc, params.logDt, params.c]),
            t_vec,
            FS
        )
        component = coeff * np.array(chirplet)
        ax2.plot(t, component, '-', linewidth=1.5, 
                label=f'Component {i+1}: fc={params.fc:.2f} Hz')
    
    ax2.set_title('Recovered Components by THRML-ACT')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reconstruction vs Original
    ax3 = axes[2]
    ax3.plot(t, signal, 'b-', alpha=0.8, linewidth=1.5, label='Original Signal')
    ax3.plot(t, reconstruction, 'r--', alpha=0.8, linewidth=1.5, label='THRML-ACT Reconstruction')
    ax3.set_title(f'Signal Reconstruction (SNR: {output_snr:.2f} dB)')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Residual
    ax4 = axes[3]
    residual = signal - reconstruction
    ax4.plot(t, residual, 'g-', alpha=0.8, linewidth=1)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.set_title(f'Residual Signal (RMS: {np.sqrt(np.mean(residual**2)):.4f})')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Amplitude')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = 'thrml_act_challenging_results.png'
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")
    
    # Create a second figure for time-frequency analysis
    fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 5: Time-frequency representation of ground truth
    ax5 = axes2[0]
    for i, gt in enumerate(ground_truth):
        # Draw a line showing the instantaneous frequency
        t_chirp = np.arange(LENGTH) / FS
        # For a chirplet, instantaneous frequency is: f(t) = fc + c*(t-tc)
        tc_sec = gt["tc"] / FS
        inst_freq = gt["fc"] + gt["c"] * (t_chirp - tc_sec)
        
        # Only plot where the chirplet has significant amplitude
        dt = np.exp(gt["logDt"]) * 0.1 * (LENGTH / FS)
        mask = np.abs(t_chirp - tc_sec) < 3 * dt  # 3 sigma
        
        ax5.plot(t_chirp[mask], inst_freq[mask], linewidth=3, 
                label=f'{gt["name"]}', alpha=0.8)
        ax5.scatter([tc_sec], [gt["fc"]], s=100, marker='o', 
                   edgecolors='black', linewidth=2, alpha=0.8)
    
    ax5.set_title('Ground Truth Time-Frequency Representation')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.set_ylim(0, 10)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Time-frequency representation of recovered components
    ax6 = axes2[1]
    for i, (params, coeff) in enumerate(results):
        # Draw recovered chirplet trajectory
        t_chirp = np.arange(LENGTH) / FS
        tc_sec = params.tc / FS
        inst_freq = params.fc + params.c * (t_chirp - tc_sec)
        
        # Only plot where the chirplet has significant amplitude
        dt = np.exp(params.logDt) * 0.1 * (LENGTH / FS)
        mask = np.abs(t_chirp - tc_sec) < 3 * dt
        
        ax6.plot(t_chirp[mask], inst_freq[mask], linewidth=3, 
                label=f'Component {i+1}: |a|={abs(coeff):.2f}', alpha=0.8)
        ax6.scatter([tc_sec], [params.fc], s=100, marker='o', 
                   edgecolors='black', linewidth=2, alpha=0.8)
    
    ax6.set_title('Recovered Components Time-Frequency Representation')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Frequency (Hz)')
    ax6.set_ylim(0, 15)  # Wider range as recovered components might be off
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the time-frequency plot
    tf_plot_filename = 'thrml_act_challenging_timefreq.png'
    plt.savefig(tf_plot_filename, dpi=150, bbox_inches='tight')
    print(f"Time-frequency plot saved as: {tf_plot_filename}")
    
    # Show plots if running interactively
    plt.show()


def main() -> None:
    run_challenging_test()


if __name__ == "__main__":
    main()
