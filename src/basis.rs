//! Scalar cubic B-spline and compactly supported spline-wavelet basis.

#[derive(Debug, Clone, Copy, Default)]
pub struct CubicBSpline;

impl CubicBSpline {
    pub const ORDER: usize = 4;
    pub const SUPPORT_START: f64 = 0.0;
    pub const SUPPORT_END: f64 = 4.0;

    #[inline]
    pub fn value(x: f64) -> f64 {
        if !x.is_finite() || !(Self::SUPPORT_START..=Self::SUPPORT_END).contains(&x) {
            return 0.0;
        }

        let mut sum = 0.0;
        for (shift, coefficient) in [
            (0.0, 1.0),
            (1.0, 0.0 - 4.0),
            (2.0, 6.0),
            (3.0, 0.0 - 4.0),
            (4.0, 1.0),
        ] {
            let t = x - shift;
            if t > 0.0 {
                sum += coefficient * t * t * t;
            }
        }

        sum / 6.0
    }

    #[inline]
    pub fn scaled(j: u32, k: i64, x: f64) -> f64 {
        let scale = 2.0_f64.powi(j as i32);
        scale.sqrt() * Self::value(scale * x - k as f64)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CubicBSplineWavelet;

impl CubicBSplineWavelet {
    pub const ORDER: usize = 4;
    pub const SUPPORT_START: f64 = 0.0;
    pub const SUPPORT_END: f64 = 7.0;

    const MASK: [f64; 11] = [
        1.0,
        0.0 - 124.0,
        1677.0,
        0.0 - 7904.0,
        18482.0,
        0.0 - 24264.0,
        18482.0,
        0.0 - 7904.0,
        1677.0,
        0.0 - 124.0,
        1.0,
    ];

    #[inline]
    pub fn value(x: f64) -> f64 {
        if !x.is_finite() || !(Self::SUPPORT_START..=Self::SUPPORT_END).contains(&x) {
            return 0.0;
        }

        let mut sum = 0.0;
        for (shift, coefficient) in Self::MASK.iter().copied().enumerate() {
            sum += coefficient * CubicBSpline::value(2.0 * x - shift as f64);
        }

        sum / 40320.0
    }

    #[inline]
    pub fn scaled(j: u32, k: i64, x: f64) -> f64 {
        let scale = 2.0_f64.powi(j as i32);
        scale.sqrt() * Self::value(scale * x - k as f64)
    }

    pub const fn mask() -> &'static [f64; 11] {
        &Self::MASK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_bspline_has_expected_support() {
        assert_eq!(CubicBSpline::value(-0.1), 0.0);
        assert_eq!(CubicBSpline::value(4.1), 0.0);
        assert!((CubicBSpline::value(2.0) - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn cubic_bspline_partitions_unity() {
        for i in 0..1000 {
            let x = -2.0 + i as f64 / 137.0;
            let sum: f64 = (-4..=5)
                .map(|k| CubicBSpline::value(x - k as f64))
                .sum();
            assert!((sum - 1.0).abs() < 1e-12, "x={x}, sum={sum}");
        }
    }

    #[test]
    fn cubic_wavelet_has_expected_support() {
        assert_eq!(CubicBSplineWavelet::value(-0.1), 0.0);
        assert_eq!(CubicBSplineWavelet::value(7.1), 0.0);
    }

    #[test]
    fn cubic_wavelet_mask_is_symmetric() {
        let mask = CubicBSplineWavelet::mask();
        for i in 0..mask.len() {
            assert_eq!(mask[i], mask[mask.len() - 1 - i]);
        }
    }
}
