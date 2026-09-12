"""chappie.solver — spectral and finite-difference PDE solvers."""

from chappie.solver.boundary import absorbing_mask
from chappie.solver.finite_difference import laplacian_fd
from chappie.solver.spectral import SpectralOps
from chappie.solver.integrators import (
    rk4_step, imex_step, suggest_dt, evolve, evolve_safe, coupled_rhs,
)
