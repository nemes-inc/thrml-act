import jax.numpy as jnp

from act import (
  ParameterRanges,
  THRMLChirpletOptimizer,
  create_test_signal,
  generate_chirplet,
)
from visualization import plot_act_decomposition

# Generate signal
signal = create_test_signal(length=512, sampling_rate=256.0)

# Fixed optimizer
optimizer = THRMLChirpletOptimizer(signal, sampling_rate=256.0)
optimizer.param_ranges = ParameterRanges(
  tc_min=100.0,  # Avoid edges (in samples: 100-400 out of 512)
  tc_max=400.0,
  fc_min=5.0,  # Low frequencies for the test signal
  fc_max=50.0,
  logDt_min=-1.5,  # Duration range
  logDt_max=0.5,
  c_min=-10.0,
  c_max=10.0,
)

# Optimize (reduced for macOS)
results = optimizer.optimize(
  n_components=3, use_coarse_sampling=True, n_refinement_candidates=5
)

print("Optimized parameters:")
for i, (params, corr) in enumerate(results):
  print(
    f"C{i + 1}: tc={params.tc:.2f}, fc={params.fc:.2f}, logDt={params.logDt:.2f}, c={params.c:.2f}, corr={corr:.4f}"
  )

# Extract components
t = jnp.arange(len(signal)) / 256.0
components = []
residual = signal.copy()

for params, _ in results:
  chirp = generate_chirplet(
    jnp.array([params.tc, params.fc, params.logDt, params.c]), t, 256.0
  )
  coeff = jnp.dot(residual, chirp)
  comp = coeff * chirp
  components.append(comp)
  residual = residual - comp

# Visualize
plot_act_decomposition(
  signal, components, [r[0] for r in results], residual, sampling_rate=256.0
)
