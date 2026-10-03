use crate::{Domain, Point, PolynomialMap};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SampleError {
    pub max_norm: f64,
    pub rmse: f64,
}

impl SampleError {
    pub fn from_samples(
        map: &PolynomialMap,
        samples: &[(Vec<f64>, Vec<f64>)],
    ) -> Result<Self, ErrorBoundError> {
        if samples.is_empty() { return Err(ErrorBoundError::EmptySamples); }
        let mut squared_sum = 0.0;
        let mut max_norm = 0.0;

        for (u, target) in samples {
            if target.len() != map.output_dimension() {
                return Err(ErrorBoundError::DimensionMismatch);
            }
            let predicted = map.evaluate(u).map_err(ErrorBoundError::Map)?;
            let mut squared = 0.0;
            for (a, b) in predicted.iter().zip(target) {
                let error = a - b;
                squared += error * error;
            }
            let norm = squared.sqrt();
            max_norm = max_norm.max(norm);
            squared_sum += squared;
        }

        Ok(Self {
            max_norm,
            rmse: (squared_sum / samples.len() as f64).sqrt(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoverageErrorBound {
    pub sample_error: f64,
    pub function_lipschitz: f64,
    pub polynomial_lipschitz: f64,
    pub coverage_radius: f64,
    pub bound: f64,
}

impl CoverageErrorBound {
    pub fn new(
        sample_error: f64,
        function_lipschitz: f64,
        polynomial_lipschitz: f64,
        coverage_radius: f64,
    ) -> Result<Self, ErrorBoundError> {
        let values = [sample_error, function_lipschitz, polynomial_lipschitz, coverage_radius];
        if values.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(ErrorBoundError::InvalidBoundParameter);
        }

        Ok(Self {
            sample_error,
            function_lipschitz,
            polynomial_lipschitz,
            coverage_radius,
            bound: sample_error
                + (function_lipschitz + polynomial_lipschitz) * coverage_radius,
        })
    }
}

pub fn coverage_radius(
    samples: &[Point],
    query_points: &[Point],
) -> Result<f64, ErrorBoundError> {
    if samples.is_empty() || query_points.is_empty() {
        return Err(ErrorBoundError::EmptySamples);
    }

    let dimension = samples[0].len();
    if dimension == 0
        || samples.iter().any(|p| p.len() != dimension)
        || query_points.iter().any(|p| p.len() != dimension)
    {
        return Err(ErrorBoundError::DimensionMismatch);
    }

    let mut radius = 0.0;
    for query in query_points {
        let nearest = samples
            .iter()
            .map(|sample| euclidean_distance(sample, query))
            .fold(f64::INFINITY, f64::min);
        radius = radius.max(nearest);
    }
    Ok(radius)
}

pub fn polynomial_lipschitz_bound(
    map: &PolynomialMap,
    domain: &Domain,
) -> Result<f64, ErrorBoundError> {
    if domain.dimension() != map.parameter_dimension() {
        return Err(ErrorBoundError::DimensionMismatch);
    }

    let mut squared_sum = 0.0;

    for parameter in 0..map.parameter_dimension() {
        let mut derivative_bound = vec![0.0; map.output_dimension()];

        for (index, coefficients) in map.terms() {
            let exponent = index.exponents()[parameter];
            if exponent == 0 { continue; }

            let mut monomial_bound = exponent as f64;
            for (j, &power) in index.exponents().iter().enumerate() {
                let (min, max) = domain.bounds()[j];
                let coordinate_bound = min.abs().max(max.abs());

                if j == parameter {
                    if power > 1 {
                        monomial_bound *= coordinate_bound.powi((power - 1) as i32);
                    }
                } else {
                    monomial_bound *= coordinate_bound.powi(power as i32);
                }
            }

            for (q, coefficient) in coefficients.iter().enumerate() {
                derivative_bound[q] += coefficient.abs() * monomial_bound;
            }
        }

        squared_sum += derivative_bound.iter().map(|x| x * x).sum::<f64>();
    }

    Ok(squared_sum.sqrt())
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

#[derive(Debug, Clone, PartialEq)]
pub enum ErrorBoundError {
    EmptySamples,
    DimensionMismatch,
    InvalidBoundParameter,
    Map(crate::PolynomialMapError),
}

impl fmt::Display for ErrorBoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "at least one sample and query point are required"),
            Self::DimensionMismatch => write!(f, "sample dimensions do not match"),
            Self::InvalidBoundParameter => write!(f, "bound parameters must be finite and non-negative"),
            Self::Map(error) => write!(f, "polynomial map error: {error}"),
        }
    }
}

impl std::error::Error for ErrorBoundError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultiIndex, PolynomialMap};

    #[test]
    fn measures_zero_training_error() {
        let mut map = PolynomialMap::new(1, 1).unwrap();
        map.add_term(MultiIndex::new(vec![0]), vec![1.0]).unwrap();
        map.add_term(MultiIndex::new(vec![1]), vec![2.0]).unwrap();

        let samples = vec![(vec![0.0], vec![1.0]), (vec![2.0], vec![5.0])];
        let error = SampleError::from_samples(&map, &samples).unwrap();

        assert!(error.max_norm < 1e-12);
        assert!(error.rmse < 1e-12);
    }

    #[test]
    fn computes_coverage_radius() {
        let samples = vec![vec![0.0], vec![1.0]];
        let queries = vec![vec![0.25], vec![0.75], vec![1.0]];
        let radius = coverage_radius(&samples, &queries).unwrap();
        assert!((radius - 0.25).abs() < 1e-12);
    }

    #[test]
    fn bounds_polynomial_lipschitz_constant() {
        let domain = Domain::new(vec![(-1.0, 1.0)]).unwrap();
        let mut map = PolynomialMap::new(1, 1).unwrap();
        map.add_term(MultiIndex::new(vec![1]), vec![2.0]).unwrap();
        map.add_term(MultiIndex::new(vec![2]), vec![3.0]).unwrap();

        let bound = polynomial_lipschitz_bound(&map, &domain).unwrap();
        assert!((bound - 8.0).abs() < 1e-12);
    }

    #[test]
    fn constructs_coverage_bound() {
        let bound = CoverageErrorBound::new(0.1, 2.0, 3.0, 0.2).unwrap();
        assert!((bound.bound - 1.1).abs() < 1e-12);
    }
}
