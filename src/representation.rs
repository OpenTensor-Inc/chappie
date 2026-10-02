//! Finite CHCS descriptions of continuous scalar functions.

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
        Self { level, coefficients: Vec::new() }
    }

    pub fn push(&mut self, index: i64, value: f64) {
        if value == 0.0 {
            return;
        }
        if let Some(existing) = self.coefficients.iter_mut().find(|c| c.index == index) {
            existing.value += value;
            if existing.value == 0.0 {
                self.coefficients.retain(|c| c.index != index);
            }
        } else {
            self.coefficients.push(SparseCoefficient { index, value });
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.coefficients.iter()
            .map(|c| c.value * CubicBSplineWavelet::scaled(self.level, c.index, x))
            .sum()
    }
}

#[derive(Debug, Clone, Default)]
pub struct ContinuousObject {
    pub base_level: u32,
    pub base: Vec<SparseCoefficient>,
    pub details: Vec<RefinementLevel>,
}

impl ContinuousObject {
    pub fn new(base_level: u32) -> Self {
        Self { base_level, base: Vec::new(), details: Vec::new() }
    }

    pub fn add_base(&mut self, index: i64, value: f64) {
        if value == 0.0 {
            return;
        }
        if let Some(existing) = self.base.iter_mut().find(|c| c.index == index) {
            existing.value += value;
            if existing.value == 0.0 {
                self.base.retain(|c| c.index != index);
            }
        } else {
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
        if !detail.coefficients.is_empty() {
            self.details.push(detail);
            self.details.sort_unstable_by_key(|d| d.level);
        }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        let base = self.base.iter()
            .map(|c| c.value * CubicBSpline::scaled(self.base_level, c.index, x))
            .sum::<f64>();
        base + self.details.iter().map(|d| d.evaluate(x)).sum::<f64>()
    }

    pub fn threshold(&mut self, threshold: f64) {
        assert!(threshold >= 0.0 && threshold.is_finite());
        for level in &mut self.details {
            level.coefficients.retain(|c| c.value.abs() >= threshold);
        }
        self.details.retain(|level| !level.coefficients.is_empty());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyadicCoordinate {
    pub depth: u32,
    pub index: u64,
}

impl DyadicCoordinate {
    pub fn encode(x: f64, depth: u32) -> Option<Self> {
        if !x.is_finite() || !(0.0..=1.0).contains(&x) || depth >= 63 {
            return None;
        }
        let cells = 1_u64 << depth;
        let index = if x == 1.0 {
            cells - 1
        } else {
            (x * cells as f64).floor() as u64
        };
        Some(Self { depth, index })
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
        2.0_f64.powi(0i32 - self.depth as i32)
    }

    pub fn midpoint_error_bound(self) -> f64 {
        self.cell_width() * 0.5
    }

    pub fn refine(self, next_bit: bool) -> Option<Self> {
        if self.depth >= 62 {
            return None;
        }
        Some(Self {
            depth: self.depth + 1,
            index: (self.index << 1) | u64::from(next_bit),
        })
    }
}
