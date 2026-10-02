//! Cubic B-spline and compactly-supported order-4 B-spline wavelet.

/// Cubic cardinal B-spline N_4.
///
/// The cardinal B-spline of order 4 is
///
/// N_4(x) = 1/6 * sum_{k=0}^4 (-1)^k C(4,k) (x-k)_+^3.
///
/// Its support is [0, 4] and it is C^2.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubicBSpline;

impl CubicBSpline {
    pub const ORDER: usize = 4;
    pub const SUPPORT_START: f64 = 0.0;
    pub const SUPPORT_END: f64 = 4.0;

    #[inline]
    pub fn value(x: f64) -> f64 {
        if !(Self::SUPPORT_START..=Self::SUPPORT_END).contains(&x) {
            return 0.0;
        }

        let terms = [
            (0.0, 1.0),
            (1.0, -4.0),
            (2.0, 6.0),
            (3.0, -4.0),
            (4.0, 1.0),
        ];

        let mut sum = 0.0;
        for (shift, coefficient) in terms {
            let t = x - shift;
            if t > 0.0 {
                sum += coefficient * t.powi(3);
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

/// Compactly-supported Chui-Wang B-spline wavelet of order 4.
///
/// The two-scale form is
///
/// psi(x) = 1/40320 *
///   [N4(2x) - 124 N4(2x-1) + 1677 N4(2x-2)
///    - 7904 N4(2x-3) + 18482 N4(2x-4)
///    - 24264 N4(2x-5) + 18482 N4(2x-6)
///    - 7904 N4(2x-7) + 1677 N4(2x-8)
///    - 124 N4(2x-9) + N4(2x-10)].
///
/// Its support is [0, 7].
#[derive(Debug, Clone, Copy, Default)]
pub struct CubicBSplineWavelet;

impl CubicBSplineWavelet {
    pub const ORDER: usize = 4;
    pub const SUPPORT_START: f64 = 0.0;
    pub const SUPPORT_END: f64 = 7.0;

    const MASK: [f64; 11] = [
        1.0,
        -124.0,
        1677.0,
        -7904.0,
        18482.0,
        -24264.0,
        18482.0,
        -7904.0,
        1677.0,
        -124.0,
        1.0,
    ];

    #[inline]
    pub fn value(x: f64) -> f64 {
        if !(Self::SUPPORT_START..=Self::SUPPORT_END).contains(&x) {
            return 0.0;
        }

        let mut sum = 0.0;
        for (j, coefficient) in Self::MASK.iter().copied().enumerate() {
            sum += coefficient * CubicBSpline::value(2.0 * x - j as f64);
        }
        sum / 40320.0
    }

    #[inline]
    pub fn scaled(j: u32, k: i64, x: f64) -> f64 {
        let scale = 2.0_f64.powi(j as i32);
        scale.sqrt() * Self::value(scale * x - k as f64)
    }

    pub fn mask() -> &'static [f64; 11] {
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
    }

    #[test]
    fn cubic_bspline_is_partition_of_unity_on_interior() {
        for i in 0..100 {
            let x = 0.25 + i as f64 / 100.0;
            let base = x.floor() as i64;
            let sum: f64 = (-3..=4)
                .map(|offset| CubicBSpline::value(x - (base + offset) as f64))
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
