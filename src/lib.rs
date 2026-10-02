//! CHCS: Continuous Hyper-Compressed Structure core.
//!
//! This crate implements the first mathematical layer of CHCS:
//! cubic B-spline scaling functions, compactly-supported Chui-Wang
//! B-spline wavelets of order 4, sparse multiresolution details,
//! and dyadic real-coordinate refinement.
//!
//! The implementation deliberately uses ordinary scalar values and
//! sparse one-dimensional coefficient lists. It does not use matrices,
//! tensors, gradient descent, or neural-network state.

pub mod basis;
pub mod representation;

pub use basis::{CubicBSpline, CubicBSplineWavelet};
pub use representation::{
    ContinuousObject, DyadicCoordinate, RefinementLevel, SparseCoefficient,
};
