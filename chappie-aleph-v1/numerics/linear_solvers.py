# chappie_aleph/numerics/linear_solvers.py
import jax
import jax.numpy as jnp
from jax import lax
from config import SpatialConfig

# ---------------------------------------------------------
# Parallel Cyclic Reduction (PCR) for Tridiagonal Systems
# Solves: a_i * x_{i-1} + b_i * x_i + c_i * x_{i+1} = d_i
# برای دیفیوژن: (I - dt * L) V_new = V_old
# L = Laplacian -> Tridiagonal: [-D/dx^2, 1+2D/dx^2, -D/dx^2]
# ---------------------------------------------------------

def thomas_forward(a, b, c, d):
    """Serial Thomas (برای CPU / مرجع تست). a, b, c, d: (N,)"""
    N = b.shape[0]
    c_prime = jnp.zeros_like(c)
    d_prime = jnp.zeros_like(d)
    
    c_prime = c_prime.at[0].set(c[0] / b[0])
    d_prime = d_prime.at[0].set(d[0] / b[0])
    
    # Scan loop (JAX lax.scan)
    def step(i, carry):
        cp, dp = carry
        denom = b[i] - a[i] * cp[i-1]
        cp = cp.at[i].set(c[i] / denom)
        dp = dp.at[i].set((d[i] - a[i] * dp[i-1]) / denom)
        return cp, dp
    
    c_prime, d_prime = lax.fori_loop(1, N, step, (c_prime, d_prime))
    
    # Back substitution
    x = jnp.zeros_like(d)
    x = x.at[N-1].set(d_prime[N-1])
    
    def back_step(i, x_val):
        return x_val.at[i].set(d_prime[i] - c_prime[i] * x_val[i+1])
    
    x = lax.fori_loop(N-2, -1, back_step, x)
    return x

# ---------------------------------------------------------
# Parallel Cyclic Reduction (PCR) - GPU Friendly
# مرجع: "Parallel Cyclic Reduction" (Hockney 1965) یا Implementations در CUDA SDK
# اینجا نسخه JAX/LAX (Log N Steps)
# ---------------------------------------------------------

def pcr_tridiagonal_solve(a, b, c, d):
    """
    a, b, c, d: (N,) arrays. 
    a[0] and c[-1] are ignored (boundaries).
    Assumes Periodic or Dirichlet/Neumann handled via BC modification of a,b,c,d.
    Returns x: (N,)
    """
    N = b.shape[0]
    
    # Padding to power of 2 for simplicity in PCR (or handle arbitrary N)
    # For now assume N is power of 2 or use generic PCR logic.
    # Generic PCR implementation in JAX is verbose.
    # برای MVP: از jax.scipy.linalg.solve_triangular اگر ماتریس کوچک است؟
    # خیر، N=8192 بزرگ است.
    
    # استفاده از lax.custom_linear_solve یا نوشتن PCR دستی.
    # اینجا یک اسکلت PCR می‌نویسیم:
    
    # Step 1: Forward Reduction (Log N stages)
    # در هر مرحله، سیستم به نیمی کاهش می‌یابد.
    
    # برای سادگی در v1، اگر GPU داریم از `jax.lax.linalg.tridiagonal_solve` (اگر اضافه شده) یا 
    # یک کالِبک به CUDA Kernel (بهترین) استفاده می‌کنیم.
    # در اینجا یک پیاده‌سازی Pure JAX ساده (O(N) با Scan) برای صحت عددی می‌نویسیم،
    # و در نسخه نهایی با Triton/CUDA Kernel جایگزین می‌شود.
    
    # JAX 0.4.23+ has `jax.lax.linalg.tridiagonal_solve` ? 
    # فعلا با Scan پیاده‌سازی می‌کنیم (کندتر از PCR اما درست).
    
    return thomas_forward(a, b, c, d) # Placeholder: Serial on CPU, Compiled to sequential on GPU (Bad perf)
    
    # TODO: Replace with `pcr_kernel` implemented in Triton/Pallas and wrapped via `jax.pure_callback` or `jax.ffi`.

def build_implicit_system(V_old: jax.Array, dt: float, D_field: jax.Array, config: SpatialConfig):
    """
    ساخت دیاگنال‌های سیستم (I - dt * L) V_new = V_old
    L = Diffusion Operator (Central Diff)
    L_ii = -2D/dx^2, L_i,i+1 = D/dx^2 (approx, assuming constant D locally)
    برای D متغیر: بهتر است اپراتور را در قالب Conservative بسازیم.
    """
    dx = config.dx
    N = V_old.shape[0]
    
    # Coefficients for (I - alpha * L) where alpha = dt
    # Main Diag: 1 + 2*alpha*D/dx^2
    # Off Diag: -alpha*D/dx^2
    
    # D at nodes
    D = D_field
    alpha = dt
    
    main_diag = 1.0 + 2.0 * alpha * D / (dx*dx)
    off_diag = -alpha * D / (dx*dx) # lower and upper (symmetric for constant D)
    
    # Boundary Conditions: Neumann (No Flux) -> Ghost points V_{-1}=V_0, V_N=V_{N-1}
    # Modifies first and last row of matrix.
    # Row 0: (1 + alpha*D/dx^2) V_0 - alpha*D/dx^2 V_1 = V_0_old
    # Row N-1: -alpha*D/dx^2 V_{N-2} + (1 + alpha*D/dx^2) V_{N-1} = V_{N-1}_old
    
    main_diag = main_diag.at[0].set(1.0 + alpha * D[0] / (dx*dx))
    main_diag = main_diag.at[-1].set(1.0 + alpha * D[-1] / (dx*dx))
    
    lower_diag = off_diag # a_i (i=1..N-1)
    upper_diag = off_diag # c_i (i=0..N-2)
    
    # Set boundaries to 0 for off-diagonals (since no neighbor)
    lower_diag = lower_diag.at[0].set(0.0)
    upper_diag = upper_diag.at[-1].set(0.0)
    
    rhs = V_old
    
    return lower_diag, main_diag, upper_diag, rhs

def implicit_diffusion_step(V_old: jax.Array, dt: float, D_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """One Implicit Diffusion Step (Crank-Nicolson style needs 0.5*dt)"""
    a, b, c, d = build_implicit_system(V_old, dt, D_field, config)
    V_new = pcr_tridiagonal_solve(a, b, c, d)
    return V_new

def solve_implicit_system(rhs: jax.Array, alpha: float, D_field: jax.Array, config: SpatialConfig) -> jax.Array:
    """Solves (I - alpha * L) V = rhs"""
    dx = config.dx
    N = rhs.shape[0]
    D = D_field
    
    main_diag = 1.0 + 2.0 * alpha * D / (dx*dx)
    off_diag = -alpha * D / (dx*dx)
    
    # BC Neumann
    main_diag = main_diag.at[0].set(1.0 + alpha * D[0] / (dx*dx))
    main_diag = main_diag.at[-1].set(1.0 + alpha * D[-1] / (dx*dx))
    lower_diag = off_diag.at[0].set(0.0)
    upper_diag = off_diag.at[-1].set(0.0)
    
    from numerics.linear_solvers import pcr_tridiagonal_solve
    return pcr_tridiagonal_solve(lower_diag, main_diag, upper_diag, rhs)