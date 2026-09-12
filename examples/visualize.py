"""3D volume visualization of the Phi, V, M fields during generation.

Renders stacked transparent slices (pseudo-volume rendering) of the CLEFPE
field Phi, the Hodgkin-Huxley potential deviation V - V_rest, and the first
memory field M0 after every generated token, plus an energy-vs-token curve.

Run: python examples/visualize.py [--grid 16] [--steps 8] [--tokens 8]
Output: PNG files in examples/output/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from chappie import Chappie, ChappieConfig
from chappie.core.config import GridConfig
from chappie.core.state import init_rest
from chappie.solver.integrators import evolve
from chappie.decoding.sampler import sample as sample_token


# ---------------------------------------------------------------------------
# Generation loop with state capture
# ---------------------------------------------------------------------------

def run_generation(model, prompt, n_tokens, key):
    """Drive autoregressive generation, returning (tokens, states).

    Mirrors chappie.inference.generate but keeps the full ChappieState after
    each generated token so the fields can be visualized.
    """
    cfg = model.config
    grid = model.grid
    spectral = model.spectral
    basis_p, dec_p = model.params.as_tuple()
    n_steps = cfg.solver.n_steps_per_token

    state = init_rest(grid, cfg)

    # Encode the prompt (no learning force)
    for tok in model.encode(prompt):
        src = model.basis.bump(int(tok), basis_p)
        state = evolve(state, src, jnp.zeros_like(state.phi), cfg, spectral,
                       n_steps=n_steps)

    # Autoregressive loop with state capture
    tokens, states = [], []
    for _ in range(n_tokens):
        key, sub = jax.random.split(key)
        logits = model.decoder.logits(state, basis_p, dec_p)
        nid = int(sample_token(logits, sub, cfg.sampling.temperature,
                               cfg.sampling.top_k, cfg.sampling.top_p))
        tokens.append(model.tokenizer.itos[nid])
        src = model.basis.bump(nid, basis_p)
        state = evolve(state, src, jnp.zeros_like(state.phi), cfg, spectral,
                       n_steps=n_steps)
        states.append(state)
    return tokens, states


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def field_arrays(states, v_rest):
    """Extract (phi, v_dev, m0) numpy arrays from captured states.

    v_dev = V - V_rest so HH activity is visible around its equilibrium.
    """
    phis = np.stack([np.asarray(s.phi) for s in states])
    vdev = np.stack([np.asarray(s.v) - v_rest for s in states])
    mems = np.stack([np.asarray(s.mem[0]) for s in states])
    return phis, vdev, mems


def global_ranges(phis, vdev, mems):
    """Per-field (vmin, vmax) shared across all frames for comparability."""
    phi_rng = (-float(np.max(np.abs(phis))), float(np.max(np.abs(phis))))
    v_rng = (-float(np.max(np.abs(vdev))), float(np.max(np.abs(vdev))))
    m_rng = (0.0, float(np.max(mems))) if np.max(mems) > 0 else (0.0, 1.0)
    return phi_rng, v_rng, m_rng


def draw_volume(ax, field, cmap, norm, n_slices, voxel_mask=None,
                voxel_alpha=0.25):
    """Stacked transparent plot_surface slices + optional voxel overlay."""
    n1, n2, n3 = field.shape
    xx, yy = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

    planes = np.linspace(0, n3 - 1, n_slices).astype(int)
    for k in planes:
        zz = np.full_like(xx, k, dtype=float)
        colors = cmap(norm(field[:, :, k]))
        ax.plot_surface(xx, yy, zz, facecolors=colors, alpha=0.4,
                        rstride=1, cstride=1, shade=False, linewidth=0,
                        antialiased=False)

    if voxel_mask is not None:
        # 3D coordinate arrays (1D arrays crash this matplotlib build)
        x, y, z = np.indices((n1 + 1, n2 + 1, n3 + 1)).astype(float)
        ax.voxels(x, y, z, voxel_mask, alpha=voxel_alpha,
                  facecolor="orange", edgecolor="k", linewidth=0.1)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=25, azim=-60)


def add_colorbar(fig, ax, cmap, norm, label):
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
    cb.set_label(label)
    return cb


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def render_volume_frames(phis, vdev, mems, tokens, out_dir, n_slices=4):
    """One single-field 3D figure per (token, field):
    vol_{phi,v,m}_step_XX.png (each file shows one field at full size)."""
    phi_rng, v_rng, m_rng = global_ranges(phis, vdev, mems)
    cmaps = (plt.cm.coolwarm, plt.cm.coolwarm, plt.cm.viridis)
    norms = (Normalize(*phi_rng), Normalize(*v_rng), Normalize(*m_rng))
    names = ("phi", "v", "m")
    titles = {
        "phi": "Phi (CLEFPE field)",
        "v": "V - V_rest (HH potential deviation)",
        "m": "M0 (memory field)",
    }
    labels = {"phi": "Phi", "v": "V - V_rest", "m": "M0"}

    for t in range(len(tokens)):
        phi, v, m = phis[t], vdev[t], mems[t]
        # "Hot zone" for the Phi voxel overlay: strong positive excitation
        hot = phi >= max(0.35 * float(np.max(phi)), 0.0) if np.max(phi) > 0 \
            else np.zeros_like(phi, dtype=bool)

        for i, (field, cmap, norm, name) in enumerate(
                zip((phi, v, m), cmaps, norms, names)):
            fig = plt.figure(figsize=(7.5, 6))
            ax = fig.add_subplot(111, projection="3d")
            mask = hot if name == "phi" else None
            draw_volume(ax, field, cmap, norm, n_slices, voxel_mask=mask)
            ax.set_title(titles[name])
            add_colorbar(fig, ax, cmap, norm, labels[name])
            fig.suptitle("Token %d: %r\n(so far: %r)" %
                         (t + 1, tokens[t], "".join(tokens[:t + 1])), y=0.94)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            fig.savefig(os.path.join(out_dir, "vol_%s_step_%02d.png"
                                     % (name, t)), dpi=120)
            plt.close(fig)
    print("Saved per-token 3D frames to %s" % out_dir)


def render_overview(phis, vdev, mems, tokens, out_dir):
    """Compact contact sheet: 2D mid-plane slices, rows = Phi/V/M, cols = tokens."""
    phi_rng, v_rng, m_rng = global_ranges(phis, vdev, mems)
    cmaps = (plt.cm.coolwarm, plt.cm.coolwarm, plt.cm.viridis)
    norms = (Normalize(*phi_rng), Normalize(*v_rng), Normalize(*m_rng))
    rows = ((phis, "Phi"), (vdev, "V - V_rest"), (mems, "M0"))

    n = len(tokens)
    fig, axes = plt.subplots(3, n, figsize=(2.0 * n, 6.5), squeeze=False)
    k = phis[0].shape[2] // 2  # mid-plane
    for r, (fields, name) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            img = ax.imshow(fields[c][:, :, k].T, origin="lower",
                            cmap=cmaps[r], norm=norms[r],
                            interpolation="bilinear")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("tok %d: %r" % (c + 1, tokens[c]), fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
    fig.suptitle("Mid-plane slices (z = n3/2) after each generated token", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "overview.png"), dpi=120)
    plt.close(fig)
    print("Saved overview contact sheet to %s" % out_dir)


def render_energy_curve(phis, model, tokens, out_dir):
    """Energy-vs-token curve showing the oscillator settling between injections."""
    grid = model.grid
    e_l2 = [float(grid.l2_energy(jnp.asarray(p))) for p in phis]
    e_grad = [float(grid.grad_energy(jnp.asarray(p))) for p in phis]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, len(tokens) + 1)
    ax1.plot(x, e_l2, "o-", color="tab:blue", label="E_l2 = 1/2 int Phi^2 dV")
    ax1.set_xlabel("generated token")
    ax1.set_ylabel("E_l2", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["%d:%r" % (i, t) for i, t in zip(x, tokens)],
                        rotation=45, fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, e_grad, "s--", color="tab:red",
             label="E_grad = 1/2 int |grad Phi|^2 dV")
    ax2.set_ylabel("E_grad", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle("Field energy after each generated token")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "energy.png"), dpi=120)
    plt.close(fig)
    print("Saved energy curve to %s" % out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", type=int, default=16, help="grid size N^3")
    ap.add_argument("--steps", type=int, default=8,
                    help="training steps before visualizing")
    ap.add_argument("--tokens", type=int, default=8,
                    help="tokens to generate and visualize")
    ap.add_argument("--prompt", type=str, default="hello")
    ap.add_argument("--out", type=str, default=None,
                    help="output directory (default examples/output)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    cfg = ChappieConfig(grid=GridConfig(n1=args.grid, n2=args.grid, n3=args.grid))
    print("Building Chappie on a %d^3 grid..." % args.grid)
    model = Chappie(cfg, key=jax.random.PRNGKey(42))
    print("  Vocab: %d, Params: %d" % (len(model.tokenizer), model.param_count()))

    # Short training run so the fields carry learned structure
    if args.steps > 0:
        print("Training %d steps on the synthetic curriculum..." % args.steps)
        dataset = model.dataset
        for s in range(args.steps):
            level = min(cfg.data.max_level, 1 + s % cfg.data.max_level)
            seq = dataset.sample_batch(1, cfg.training.seq_len, level,
                                       key=1000 + s)[0]
            loss = model.train_step(seq)
        print("  final train NLL: %.4f (random = %.4f)"
              % (loss, float(np.log(len(model.tokenizer)))))

    # Generate and capture fields
    print("Generating %d tokens from %r..." % (args.tokens, args.prompt))
    tokens, states = run_generation(model, args.prompt, args.tokens,
                                    jax.random.PRNGKey(0))
    print("  generated text: %r" % ("".join(tokens)))
    phis, vdev, mems = field_arrays(states, model.config.hh.rest)

    # Render
    render_volume_frames(phis, vdev, mems, tokens, out_dir)
    render_overview(phis, vdev, mems, tokens, out_dir)
    render_energy_curve(phis, model, tokens, out_dir)
    print("Done. All figures written to %s" % out_dir)


if __name__ == "__main__":
    main()
