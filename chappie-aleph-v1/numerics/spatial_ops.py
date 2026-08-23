# chappie_aleph/numerics/spatial_ops.py
import jax
import jax.numpy as jnp
from jax import lax
from config import SpatialConfig

# ---------------------------------------------------------
# 1. Diffusion Operator: ∇·(D(x) ∇V)
# Discretization: Central Difference 2nd Order (Compact Stencil)
# (D_{i+1/2} (V_{i+1} - V_i) - D_{i-1/2} (V_i - V_{i-1})) / dx^2
# ---------------------------------------------------------
def diffusion_rhs(V: jax.Array, D_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """
    V: (N,), D_field: (N,) or (N-1,) at interfaces
    Returns: dV/dt_diffusion (N,)
    Boundary: Neumann (Zero Flux) -> Ghost cells mirror interior.
    """
    dx = config.dx
    N = V.shape[0]
    
    # D at interfaces (i+1/2) -> Harmonic mean is better for discontinuities, Arithmetic for smooth
    # D_interface = (D[:-1] + D[1:]) / 2.0 
    D_ip12 = (D_field[:-1] + D_field[1:]) * 0.5 # (N-1,)
    
    # Fluxes F = -D * dV/dx
    dV_dx = (V[1:] - V[:-1]) / dx # (N-1,)
    Flux = -D_ip12 * dV_dx        # (N-1,)
    
    # Divergence of Flux
    # dV/dt = -dF/dx = -(F_{i+1/2} - F_{i-1/2}) / dx
    # Padding for boundaries (Zero Flux -> F_{-1/2}=0, F_{N-1/2}=0)
    Flux_padded = jnp.pad(Flux, (1, 1), mode='constant', constant_values=0.0) # (N+1,)
    div_F = (Flux_padded[1:] - Flux_padded[:-1]) / dx # (N,)
    
    return div_F

# ---------------------------------------------------------
# 2. Wave/Transport Operator: Hyperbolic Part
# Option A: 1st Order Transport: dV/dt + c * dV/dx = 0  (Upwind / WENO)
# Option B: 2nd Order Wave: d2V/dt2 = c^2 * d2V/dx2     (Leapfrog / Newmark)
# ما Option A را برای اولین نسخه (First Order System) پیاده می‌کنیم.
# Upwind Scheme: اگر c > 0 -> از چپ (i-1), اگر c < 0 -> از راست (i+1)
# WENO5 برای عدم نوسان (Oscillation Free) پیشنهادی است.
# ---------------------------------------------------------
def wave_rhs_upwind(V: jax.Array, c_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """
    dV/dt = - c * dV/dx
    Upwind 1st Order (Monotone, Diffusive) -> Baseline
    """
    dx = config.dx
    # c at nodes
    c = c_field
    
    # Upwind difference
    # dV/dx_forward = (V[i+1] - V[i]) / dx
    # dV/dx_backward = (V[i] - V[i-1]) / dx
    dV_fwd = jnp.roll(V, -1) - V
    dV_bwd = V - jnp.roll(V, 1)
    
    # Boundary handling (Neumann/Dirichlet) -> roll wraps around! Must mask.
    # Better: Use lax.conv or manual slicing with padding.
    # JAX friendly slicing:
    V_pad = jnp.pad(V, (1, 1), mode='edge') # Neumann BC
    dV_fwd = (V_pad[2:] - V_pad[1:-1]) / dx
    dV_bwd = (V_pad[1:-1] - V_pad[:-2]) / dx
    
    # Upwind selection
    dV_dx = jnp.where(c >= 0, dV_bwd, dV_fwd)
    
    return -c * dV_dx

def wave_rhs_weno5(V: jax.Array, c_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """
    WENO5 Reconstruction for Flux.
    Heavy math, standard implementation. 
    برای v1 ابتدا Upwind استفاده می‌کنیم، WENO را بعداً به عنوان Kernel بهینه جایگزین می‌کنیم.
    """
    # TODO: Implement WENO5 flux splitting (Lax-Friedrichs flux + WENO recon)
    # Placeholder calling Upwind
    return wave_rhs_upwind(V, c_field, config)

# ---------------------------------------------------------
# 3. Laplacian (Spectral / FFT) - برای Solver ضمنی طیفی
# اگر مرزی دوره‌ای (Periodic) داریم، FFT سریع‌ترین است.
# ---------------------------------------------------------
def laplacian_spectral(V: jax.Array, config: SpatialConfig) -> jax.Array:
    """d2V/dx2 via FFT. Assumes Periodic BC."""
    N = config.n_nodes
    dx = config.dx
    k = jnp.fft.fftfreq(N, d=dx) * 2 * jnp.pi
    # k^2 factor
    V_hat = jnp.fft.fft(V)
    V_xx_hat = - (k**2) * V_hat
    return jnp.fft.ifft(V_xx_hat).real