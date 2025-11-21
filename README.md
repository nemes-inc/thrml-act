# THRML Chirplet Transform

Implementation combining THRML discrete sampling with gradient refinement for adaptive chirplet decomposition.

## Getting Started

> **Note:** This implementation has only been tested on Mac M series laptops. Compatibility with other platforms is not guaranteed. By default, the code runs on CPU for maximum compatibility.

```bash
# Create the virtual environment
uv venv .venv

# Sync dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Run the visualization
python run_visualization.py

# Run the test suite
python act-challenging-test.py
python act-profile-test.py
python act-synth-test.py
```

## Usage

### Basic Decomposition

```python
from act import decompose_signal

components, parameters, residual = decompose_signal(
    signal,
    sampling_rate=256.0,
    n_components=10,
    residual_threshold=0.01,
    use_thrml=True
)
```

### Configuration

```python
from act import THRMLChirpletOptimizer

optimizer = THRMLChirpletOptimizer(signal, sampling_rate=256.0)
results = optimizer.optimize(
    n_components=5,
    use_coarse_sampling=True,
    n_refinement_candidates=10
)
```

### Parameters

- `n_levels`: Discretization resolution (default: 32)
- `beta`: Sampling temperature (default: 0.5-1.0)
- `n_samples`: Number of MCMC samples (default: 5000)
- `n_warmup`: Burn-in iterations (default: 1000)
- `n_refinement_candidates`: Number of candidates for BFGS refinement (default: 10)

## API

### `decompose_signal()`
Decompose a signal into chirplet components.

**Parameters**
- `signal`: Input signal array
- `sampling_rate`: Sampling frequency (Hz)
- `n_components`: Maximum number of components
- `residual_threshold`: Stop when residual < threshold * signal energy
- `use_thrml`: Whether to use THRML sampling (vs pure BFGS)

**Returns:** `(components, parameters, residual)`

### `ChirpletParams`
Dataclass containing chirplet parameters
- `tc`: Time center (samples)
- `fc`: Frequency center (Hz)
- `logDt`: Log duration scale
- `c`: Chirp rate (Hz/s)

## Mathematical

The Adaptive Chirplet Transform uses four continuous parameters to define Gaussian chirplet atoms.

$$g_{t_c,f_c,\log\Delta t,c}(t) = \frac{1}{\sqrt{2\pi}\Delta t} \exp\left(-\frac{1}{2}\left(\frac{t-t_c}{\Delta t}\right)^2\right) \exp\left(j2\pi\left[c(t-t_c)^2 + f_c(t-t_c)\right]\right)$$

> **Note:** Some references use $\exp(-\frac{1}{2}(t/\Delta t)^2)$ for the Gaussian envelope, but this would center the atom at $t=0$ regardless of $t_c$.

**Components**
- **Gaussian envelope**: $\frac{1}{\sqrt{2\pi}\Delta t} \exp\left(-\frac{1}{2}\left(\frac{t-t_c}{\Delta t}\right)^2\right)$ provides time localization centered at $t_c$
- **Chirp modulation**: $\exp\left(j2\pi\left[c(t-t_c)^2 + f_c(t-t_c)\right]\right)$ encodes frequency sweep
  - Linear term $f_c(t-t_c)$: base frequency
  - Quadratic term $c(t-t_c)^2$: frequency chirp rate

## How It Works

The implementation uses a hybrid coarse-to-fine optimization strategy.

1. **THRML Coarse Sampling**: Explores discrete parameter space (32 levels per parameter) using probabilistic graphical model with 20 binary spin nodes. Samples 5000 configurations to find promising regions.
2. **BFGS Refinement**: Gradient-based optimization from top THRML candidates (typically 10 candidates refined with L-BFGS).
3. **Matching Pursuit**: Extracts components iteratively from residual signal until convergence threshold is met.

## When to Use THRML

**THRML provides maximum benefit when**
- Highly multimodal signals (overlapping chirps with similar characteristics)
- Noisy measurements (SNR < 10 dB) where gradient estimates are unreliable

**Skip THRML when,**
- Real-time processing (< 10ms latency) - use pure BFGS
- High-SNR signals (> 30 dB) with well-separated components

## Implementation Notes

- **Discretization**: 32 levels (5 bits) per parameter provides decent balance between resolution and tractability
- **Sampling**: Requires 500-1000 warmup iterations for convergence
- **Memory**: On-the-fly chirplet generation recommended for large signals to avoid 4GB+ GPU memory usage

## Dictionary Search Strategy Comparison

### C++ ACT (Exhaustive Search)

The reference C++ implementation uses exhaustive dictionary search

```cpp
// GEMV: scores = A^T * signal  (600k atoms × signal length)
act::blas::gemv_colmajor_trans(m, n, alpha, A, x, ...);
int best_idx = cblas_isamax(n, scores, 1);  // Find max correlation
```

### Python THRML (Probabilistic Sampling)

This implementation uses probabilistic sampling via THRML

```python
# Sample parameter space probabilistically
sampled_params = sample_chirplet_parameters(...)  # 3000 samples via MCMC
correlations = [compute_correlation(residual, generate_chirplet(p)) 
                for p in sampled_params]
top_candidates = top_k(correlations)  # Select best 10-12
```

### Performance Comparison

| Implementation | Dictionary Size | Search Method | Speedup |
|---|---|---|---|
| C++ CPU (BLAS) | 600k atoms | GEMV (all atoms) | 2-3.5× real-time (Mac M1) |
| C++ GPU (MLX) | 600k atoms | Matrix multiply | **31-54× real-time** (M1/RTX4090) |
| Python THRML | ~3k samples | Probabilistic | Not measured |

However, the **search strategy** is fundamentally different
- **C++ ACT**: Exhaustive dictionary search (optimized via BLAS/GPU)
- **Python THRML**: Probabilistic sampling (optimized via THRML+BFGS hybrid)

The Python implementation prioritizes **sample efficiency** (fewer evaluations) over **raw throughput** (GPU-accelerated exhaustive search), making it suitable for exploration and research rather than real-time processing.

## References

- THRML: <https://github.com/extropic-ai/thrml>
- ACT Implementation: <https://byron-the-bulb.github.io/act/>
