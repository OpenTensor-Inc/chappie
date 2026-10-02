//! CHCS continuous representation.
//!
//! A ContinuousObject is a finite description of a continuous function:
//!
//! F(x) = base(x) + sum_j detail_j(x)
//!
//! where the base and detail functions are built from compactly-supported
//! cubic B-spline and B-spline-wavelet atoms.

use crate::basis::{CubicBSpline, CubicBSplineWavelet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparseCoefficient {
    pub index: i64,
    pub value: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RefinementLevel {
    pub level: u32,
    pub coefficients: Vec<SparseCoefficient>,
}

impl RefinementLevel {
    pub fn new(level: u32) -> Self {
        Self {
            level,
            coefficients: Vec::new(),
        }
    }

    pub fn push(&mut self, index: i64, value: f64) {
        if value != 0.0 {
            self.coefficients.push(SparseCoefficient { index, value });
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.coefficients
            .iter()
            .map(|c| c.value * CubicBSplineWavelet::scaled(self.level, c.index, x))
            .sum()
    }
}

/// A finite continuous object.
///
/// The object is not a table of sampled values. Its coefficients define
/// a continuous function through the B-spline/wavelet basis.
#[derive(Debug, Clone, Default)]
pub struct ContinuousObject {
    pub base_level: u32,
    pub base: Vec<SparseCoefficient>,
    pub details: Vec<RefinementLevel>,
}

impl ContinuousObject {
    pub fn new(base_level: u32) -> Self {
        Self {
            base_level,
            base: Vec::new(),
            details: Vec::new(),
        }
    }

    pub fn add_base(&mut self, index: i64, value: f64) {
        if value != 0.0 {
            self.base.push(SparseCoefficient { index, value });
        }
    }

    pub fn add_detail(&mut self, level: u32, index: i64, value: f64) {
        if let Some(existing) = self.details.iter_mut().find(|d| d.level == level) {
            existing.push(index, value);
            return;
        }

        let mut detail = RefinementLevel::new(level);
        detail.push(index, value);
        self.details.push(detail);
        self.details.sort_by_key(|d| d.level);
    }

    /// Evaluate the continuous object at x.
    pub fn evaluate(&self, x: f64) -> f64 {
        let base: f64 = self
            .base
            .iter()
            .map(|c| c.value * CubicBSpline::scaled(self.base_level, c.index, x))
            .sum();

        base + self.details.iter().map(|d| d.evaluate(x)).sum::<f64>()
    }

    /// Return the maximum absolute contribution of all retained detail
    /// coefficients, using a conservative coefficient-norm bound.
    pub fn detail_l1_bound(&self) -> f64 {
        self.details
            .iter()
            .flat_map(|d| d.coefficients.iter())
            .map(|c| c.value.abs())
            .sum()
    }

    /// Remove coefficients below a threshold.
    ///
    /// This is an explicit approximation/compression operation. The caller
    /// supplies the threshold and therefore controls the representation error.
    pub fn threshold(&mut self, threshold: f64) {
        for level in &mut self.details {
            level
                .coefficients
                .retain(|coefficient| coefficient.value.abs() >= threshold);
        }
        self.details.retain(|level| !level.coefficients.is_empty());
    }
}

/// A finite dyadic description of a real coordinate in [0, 1].
///
/// k / 2^depth is the left endpoint of the dyadic cell containing x.
/// The cell width is 2^-depth, so the representation carries an explicit
/// deterministic error bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyadicCoordinate {
    pub depth: u32,
    pub index: u64,
}

impl DyadicCoordinate {
    pub fn encode(x: f64, depth: u32) -> Self {
        assert!(x.is_finite(), "x must be finite");
        assert!((0.0..=1.0).contains(&x), "x must lie in [0, 1]");
        let cells = 1_u64 << depth;
        let mut index = (x * cells as f64).floor() as u64;
        if index >= cells {
            index = cells - 1;
        }
        Self { depth, index }
    }

    pub fn lower_endpoint(self) -> f64 {
        self.index as f64 / 2.0_f64.powi(self.depth as i32)
    }

    pub fn upper_endpoint(self) -> f64 {
        (self.index + 1) as f64 / 2.0_f64.powi(self.depth as i32)
    }

    pub fn midpoint(self) -> f64 {
        (self.lower_endpoint() + self.upper_endpoint()) * 0.5
    }

    pub fn cell_width(self) -> f64 {
        2.0_f64.powi(-(self.depth as i32))
    }

    pub fn midpoint_error_bound(self) -> f64 {
        self.cell_width() * 0.5
    }

    pub fn refine(self, next_bit: bool) -> Self {
        Self {
            depth: self.depth + 1,
            index: (self.index << 1) | u64::from(next_bit),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyadic_coordinate_has_explicit_error_bound() {
        let x = 0.371829;
        let c = DyadicCoordinate::encode(x, 20);
        assert!((c.midpoint() - x).abs() <= c.midpoint_error_bound());
        assert!(c.midpoint_error_bound() <= 2f64.powi(-21));
    }

    #[test]
    fn continuous_object_is_finite() {
        let mut object = ContinuousObject::new(0);
        object.add_base(0, 1.0);
        object.add_detail(1, 2, 0.25);
        object.add_detail(2, 5, -0.125);

        let value = object.evaluate(0.5);
        assert!(value.is_finite());
        assert_eq!(object.details.len(), 2);
    }

    #[test]
    fn threshold_removes_small_details() {
        let mut object = ContinuousObject::new(0);
        object.add_detail(1, 0, 1e-4);
        object.add_detail(1, 1, 0.2);
        object.threshold(1e-3);

        assert_eq!(object.details.len(), 1);
        assert_eq!(object.details[0].coefficients.len(), 1);
    }
}
