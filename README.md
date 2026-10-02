# Chappie / CHCS

This repository currently contains the first Rust implementation of the CHCS
(Continuous Hyper-Compressed Structure) mathematical core.

## Mathematical model

CHCS represents a continuous object by a finite description

\[
F(x)=\sum_k a_k\phi_{j,k}(x)
    +\sum_{r}\sum_k d_{r,k}\psi_{r,k}(x),
\]

where:

- \(\phi\) is the cubic cardinal B-spline \(N_4\),
- \(\psi\) is the compactly-supported order-4 Chui-Wang B-spline wavelet,
- \(j\) and \(r\) are dyadic refinement levels,
- only explicitly retained coefficients are stored.

The cubic B-spline has support \([0,4]\) and is \(C^2\). The selected
order-4 compactly-supported B-spline wavelet has support \([0,7]\).

A separate dyadic coordinate representation stores

\[
q_n(x)=\frac{\lfloor 2^n x\rfloor}{2^n},
\]

for \(x\in[0,1]\), with cell width \(2^{-n}\). Its midpoint reconstruction has
the deterministic bound

\[
|x-\widehat{x}_n|\le 2^{-n-1}.
\]

This is an approximation bound, not a claim that an arbitrary real number can
be represented exactly by finite digital information.

## Architecture

The current code intentionally contains no neural-network layer, tensor,
matrix, backpropagation, or gradient-descent machinery.

The intended hierarchy is:

continuous data
-> B-spline continuous basis
-> local wavelet refinements
-> continuous object
-> manifold/chart layer
-> CAH identity
-> CIC relational structure

CAH and CIC are not implemented yet. The repository is currently at the
continuous-basis layer.

## Status

This is the mathematical foundation, not yet a complete language model.
The next implementation stage is the exact analysis/synthesis transform for
the selected B-spline wavelet basis, followed by chart/manifold structures.
