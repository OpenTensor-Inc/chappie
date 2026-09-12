# Chappie v1 — CLEFPE Field Language Model Design

**Date:** 2026-08-24  
**Status:** Implemented (PoC)  
**Spec:** §1–§33 of the Chappie design document

## 1. Overview

Chappie v1 replaces Transformer attention and stored weight matrices with 3D field dynamics on a periodic torus Ω = [0, L)³. The system couples three PDE subsystems:

1. **CLEFPE oscillator** (spec §2/§10): damped wave with nonlinear potential, HH coupling, and memory
2. **Hodgkin-Huxley fields** (spec §4): spatial V, m, h, n with classical gating rates
3. **Memory fields** (spec §3): diffusion-decay storage with coupling from Φ

## 2. Discretization

- **Grid:** N×N×N regular (default 16³), torus [0, L)³
- **Spectral Laplacian:** ∇²Φ = irfftn(−|k|² · rfftn(Φ)) — exact for periodic BCs
- **FD Laplacian:** 7-point stencil with Dirichlet/Neumann/Absorbing BCs
- **Time integration:** IMEX semi-implicit (default) or explicit RK4

### IMEX Scheme

Linear terms (c²∇²Φ, D∇²W, −γW, ρ∇²M − μM) are solved implicitly per Fourier mode via backward Euler. Nonlinear terms (HH gating, Φ³ potential, coupling) are explicit. This makes the scheme unconditionally stable for the diffusive/wave part.

Per-mode implicit solve for (Φ, W):
```
(I − dt·A) · [Φ̂, Ŵ] = [Φ̂₀, Ŵ₀ + dt·nl_ŵ]
```
where A is the 2×2 linear operator per mode.

## 3. CLEFPE Equation (§10)

First-order form:
```
Φ_t = W
W_t = c²∇²Φ + D∇²W − γW + κ(V−V₀) + Σ_s χ_s M_s + F_learning − λ_R(aΦ + bΦ³)
```

Parameters (configurable): c=1.2, D=0.05, γ=0.8, κ=0.5, V₀=−65, λ_R=0.1, a=−0.5, b=0.25

## 4. Memory Fields (§3)

```
∂t M_s = ρ_s ∇²M_s − μ_s M_s + η_s Φ
```

Two fields: short-term (ρ=0.02, μ=0.3, η=0.4) and long-term (ρ=0.005, μ=0.02, η=0.2). Coupled back to CLEFPE via χ coefficients.

## 5. Hodgkin-Huxley (§4)

Standard HH rates (dimensionless, field units):
```
α_m = 0.1(V+40) / (1 − exp(−(V+40)/10))     β_m = 4 exp(−(V+65)/18)
α_h = 0.07 exp(−(V+65)/20)                     β_h = 1 / (1 + exp(−(V+35)/10))
α_n = 0.1(V+55) / (1 − exp(−(V+55)/10))       β_n = 0.125 exp(−(V+65)/80)
```

Guarded against division-by-zero at V=−40 and V=−55 via `_x_over_1mexp`.

## 6. Sign Convention Fix (§6/§7)

The spec writes `P ∝ exp(−R/τ)` but its force formula `F = −δE/δΦ` is only consistent with `P ∝ exp(+R/τ)`:

```python
logits = R / τ + bias        # R = ⟨Φ, ψ̄_y⟩
P = softmax(logits)
F = (gain/τ)(ψ̄_{y*} − Σ_y P(y) ψ̄_y)
```

This pushes Φ toward the target basis and away from the average prediction. The alternative sign makes the model repel the target — mathematically inconsistent.

## 7. Energy Functional (§8)

```
E = E_diff + E_wave + E_HH + E_mem + E_reg
```

- E_diff = ½∫|∇Φ|² dV
- E_wave = ½∫(W² + c²|∇Φ|²) dV
- E_HH = ½C_m∫V² dV + ½∫(m²+h²+n²) dV
- E_mem = ½∫Σ_s M_s² dV
- E_reg = ∫(a/2 Φ² + b/4 Φ⁴) dV

All computed spectrally (FFT gradients). Logged for stability monitoring.

## 8. Training Regime

**Hybrid:** Force + tiny-param gradients

1. **Force rollout** (teacher-forced): analytic CLEFPE learning force drives Φ; no BPTT through the rollout
2. **Tiny-param update** (every k steps): truncated BPTT through a short unroll (n_unroll=4) for basis widths/amps and decoder bias; Adam optimizer

The force rollout uses `jax.lax.scan` (compiled once, ~0.02s/step after JIT). The tiny-param update uses `jax.grad` on the short unroll.

## 9. Synthetic Curriculum (§17)

No heavy datasets needed. Six levels of increasing complexity:

1. **Random chars** — maximum entropy baseline
2. **Bigram Markov** — predictable next-char (e.g., a→b, b→a)
3. **Repeated words** — "cat dog cat dog"
4. **Cyclic patterns** — "abc abc abc"
5. **Variable binding** — "x5 x5 y3 y3"
6. **Counting pairs** — "a1 b2 c3"

Curriculum: level increases linearly over training.

## 10. Checkpointing (§17)

`.npz` format with config YAML, tokenizer vocab, basis params, decoder bias, Adam state, step, and metrics. Load reconstructs the full model state.

## 11. Tests

57 tests covering:
- Config roundtrip (YAML)
- Grid (Laplacian accuracy)
- HH (rates bounded, rest equilibrium)
- Memory (decay without source)
- CLEFPE (energy bounded)
- Solvers (spectral vs FD, IMEX stability, CFL)
- Encoding (tokenizer roundtrip, basis projection normalization)
- Decoding (logits shape, probs sum to 1, sign convention)
- Training (loss finite, force direction, Adam correctness)
- Model (encode/decode, train step, generate, save/load)
