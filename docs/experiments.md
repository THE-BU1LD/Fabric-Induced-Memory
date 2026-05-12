# experiments.md

## Experimental Setup

### Benchmarks

The model is evaluated across multiple dynamical systems:

- Lorenz-96 (chaotic ODE)
- Diffusion equation (PDE)
- Wave equation (PDE)
- Fractional diffusion systems
- Stochastic differential equations

---

### Data Generation

Each benchmark generates trajectories:

\[
\{x_0, x_1, \dots, x_T\}
\]

Training pairs are constructed as:

\[
(x_t, x_{t+1})
\]

Evaluation uses full autoregressive rollout.

---

### Training Details

- Optimizer: AdamW
- Learning rate: \(3 \times 10^{-4}\)
- Batch size: 32
- Scheduler: cosine decay with warmup
- Gradient clipping: 1.0
- Mixed precision training enabled

---

### Loss Function

\[
\mathcal{L} =
\lambda_p \mathcal{L}_{pred} +
\lambda_m \mathcal{L}_{mem} +
\lambda_s \mathcal{L}_{stab} +
\lambda_r \mathcal{L}_{retr} +
\lambda_c \mathcal{L}_{cons}
\]

---

### Metrics

- Mean Squared Error (MSE)
- Long-horizon error growth
- Distribution alignment
- Spectral consistency
- Drift over time

---

### Evaluation Protocol

Models are evaluated via autoregressive rollout:

\[
\hat{x}_{t+1} = f(\hat{x}_t)
\]

Metrics are computed over full trajectories.

---

### Ablation Studies

We evaluate the contribution of each component:

| Variant | Description |
|--------|------------|
| Full Model | Complete FIM |
| No Memory | Removes memory module |
| No Retrieval | Disables retrieval injection |
| Low Capacity | Reduced hidden size |
| High Noise | Increased stochasticity |

---

### Results

The full model consistently outperforms ablations in:

- Stability over long horizons
- Lower accumulated error
- Better spectral fidelity
- Improved robustness to noise

---

### Visualization

We report:

- Error curves (log-scale)
- Heatmaps over time and dimension
- PCA projections
- Spectral analysis
- Trajectory comparisons

---

### Reproducibility

All experiments use fixed seeds and deterministic settings where applicable.