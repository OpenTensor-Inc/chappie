use chcs::{ContinuousObject, CubicBSpline, CubicBSplineWavelet, DyadicCoordinate};

#[test]
fn basis_and_wavelet_are_continuous_objects() {
    let x = 1.75;
    let phi = CubicBSpline::value(x);
    let psi = CubicBSplineWavelet::value(x);

    assert!(phi.is_finite());
    assert!(psi.is_finite());
}

#[test]
fn dyadic_refinement_reduces_cell_width() {
    let coarse = DyadicCoordinate::encode(0.371829, 8);
    let fine = coarse.refine(true);

    assert_eq!(fine.depth, 9);
    assert_eq!(fine.index, coarse.index * 2 + 1);
    assert!(fine.cell_width() < coarse.cell_width());
}

#[test]
fn finite_description_defines_a_function() {
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
