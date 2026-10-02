use chcs::{ContinuousObject, CubicBSpline, CubicBSplineWavelet, DyadicCoordinate};

#[test]
fn cubic_basis_is_finite() {
    for i in 0..100 {
        let x = -1.0 + i as f64 / 20.0;
        assert!(CubicBSpline::value(x).is_finite());
        assert!(CubicBSplineWavelet::value(x).is_finite());
    }
}

#[test]
fn cubic_bspline_partitions_unity() {
    for i in 0..200 {
        let x = -2.0 + i as f64 / 37.0;
        let sum: f64 = (-4..=5)
            .map(|k| CubicBSpline::value(x - k as f64))
            .sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}

#[test]
fn dyadic_coordinate_has_explicit_error_bound() {
    let x = 0.371829;
    let coordinate = DyadicCoordinate::encode(x, 20).unwrap();
    assert!((coordinate.midpoint() - x).abs() <= coordinate.midpoint_error_bound());
    assert!(coordinate.midpoint_error_bound() <= 2f64.powi(0i32 - 21));
}

#[test]
fn dyadic_refinement_reduces_cell_width() {
    let coarse = DyadicCoordinate::encode(0.371829, 8).unwrap();
    let fine = coarse.refine(true).unwrap();
    assert_eq!(fine.depth, 9);
    assert_eq!(fine.index, coarse.index * 2 + 1);
    assert!(fine.cell_width() < coarse.cell_width());
}

#[test]
fn continuous_object_defines_a_scalar_function() {
    let mut object = ContinuousObject::new(0);
    object.add_base(0, 1.0);
    object.add_base(1, 0.5);
    object.add_detail(1, 0, 0.25);
    let y0 = object.evaluate(0.25);
    let y1 = object.evaluate(0.75);
    assert!(y0.is_finite());
    assert!(y1.is_finite());
    assert_ne!(y0, y1);
}

#[test]
fn duplicate_coefficients_are_merged() {
    let mut object = ContinuousObject::new(0);
    object.add_base(2, 0.25);
    object.add_base(2, 0.75);
    assert_eq!(object.base.len(), 1);
    assert_eq!(object.base[0].value, 1.0);

    object.add_detail(1, 3, 0.25);
    object.add_detail(1, 3, 0.0 - 0.25);
    assert!(object.details.is_empty());
}
