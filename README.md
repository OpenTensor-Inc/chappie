# Chappie v1 — CLEFPE Field Language Model

A language model using **3D PDE field dynamics** instead of Transformers.

Chappie evolves coupled fields on a 3D torus:
- **Φ** (CLEFPE oscillator): damped wave with nonlinear potential and memory coupling
- **V, m, h, n** (Hodgkin-Huxley): field-based spiking dynamics
- **M₁, M₂** (memory fields): short-term and long-term diffusion-decay storage

Tokens become 3D bump excitations; readout is spectral projection; the **learning force** is an analytic functional gradient (no autodiff through the PDE for the core dynamics).

## Architecture (§33 Pipeline)

```
Token → 3D bump excitation → PDE evolution (CLEFPE + HH + memory)
    → spectral projection → logits → softmax → next token
```

**No attention, no dense weight matrices, no embedding lookup.** The few learnable scalars (basis widths/amps, decoder bias) are updated via a short unroll of the PDE (truncated BPTT). The field itself is the computation, not stored weights.

## Quick Start

```bash
pip install -e ".[dev]"

# Run all tests (57 tests)
python -m pytest tests/ -v

# Train on synthetic curriculum
python -m chappie.cli train --steps 50 --grid 12

# Generate text
python -m chappie.cli generate --ckpt artifacts/ckpt/final.npz --prompt "hello"

# Visualize Phi / V / M fields during generation (3D volume PNGs)
python examples/visualize.py
```

## Example

```python
from chappie import Chappie

model = Chappie()                           # default config (16³ grid)
text = model.generate("the ", max_new_tokens=30)
loss = model.train_step(model.encode("hello world"))
model.save("ckpt.npz")
```

## Module Layout

```
chappie/
  core/           CLEFPE, HH, memory, grid, state, energy
  solver/         spectral (FFT), finite-difference, IMEX/RK4 integrators
  encoding/       tokenizer (char/BPE), basis (Gaussian/Fourier/Wavelet)
  decoding/       field decoder, sampler (top-k/top-p)
  training/       loss, learning force, Adam optimizer, trainer, synthetic dataset
  inference/      autoregressive generation
  config/         YAML config + loader (no PyYAML dependency)
  cli/            train, generate, eval subcommands
tests/            57 unit tests covering all modules
```

## Equations (spec §10)

```
∂t Φ   = W
∂t W   = c²∇²Φ + D∇²W − γW + κ(V−V₀) + Σχ_s M_s + F_learning − λ_R(aΦ + bΦ³)
∂t M_s = ρ_s ∇²M_s − μ_s M_s + η_s Φ
C_m ∂t V  = I_input + ξΦ − g_Na m³h(V−E_Na) − g_K n⁴(V−E_K) − g_L(V−E_L)
∂t m  = α_m(V)(1−m) − β_m(V)m      (and similarly for h, n)
```

## Learning Force (spec §7)

```
F_learning = (gain/τ) · (ψ̄_{y*} − Σ_y P(y) ψ̄_y)
```

A contrastive field force: pushes Φ toward the target token's basis function and away from the current average prediction. Active only during teacher-forced training.

## Sign Convention (§6)

The spec's `P ∝ exp(−R/τ)` is inconsistent with its own force formula. Chappie uses `logits = R/τ + b` (i.e. `P ∝ exp(+R/τ)`), which is the unique choice consistent with the CLEFPE learning force derivation. This is documented in the design spec.

## Performance

- **57 tests pass** on CPU in ~42s
- Grid 8³ for tests, 16³ for training demos
- First JIT compile ~13s; subsequent steps ~0.02s
- ~12K stored params (vs ~165K for a comparable char-RNN)

## Config

Default config lives at `chappie/config/default.yaml`. All parameters are configurable via YAML or CLI overrides:

```bash
python -m chappie.cli train --grid 12 --steps 100 --lr 0.001
```

## Scientific Correctness (§31)

This is a **proof-of-concept** research prototype. It does not claim to outperform production LLMs. All assumptions (discretization, stability, sign convention) are documented in the design spec.

## License

GPL-3.0-or-later
