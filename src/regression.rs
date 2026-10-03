use crate::{MultiIndex, PolynomialMap, PolynomialMapError};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegressionConfig {
    pub degree: usize,
}

impl RegressionConfig {
    pub fn new(degree: usize) -> Self {
        Self { degree }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegressionError {
    EmptySamples,
    ParameterDimensionMismatch { expected: usize, actual: usize },
    OutputDimensionMismatch { expected: usize, actual: usize },
    NonFiniteSample,
    Underdetermined { samples: usize, terms: usize },
    SingularSystem,
    Map(PolynomialMapError),
}

impl fmt::Display for RegressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "regression requires at least one sample"),
            Self::ParameterDimensionMismatch { expected, actual } =>
                write!(f, "parameter dimension {actual}, expected {expected}"),
            Self::OutputDimensionMismatch { expected, actual } =>
                write!(f, "output dimension {actual}, expected {expected}"),
            Self::NonFiniteSample => write!(f, "samples must contain only finite values"),
            Self::Underdetermined { samples, terms } =>
                write!(f, "regression is underdetermined: {samples} samples for {terms} polynomial terms"),
            Self::SingularSystem => write!(f, "regression system is singular"),
            Self::Map(error) => write!(f, "polynomial map error: {error}"),
        }
    }
}

impl std::error::Error for RegressionError {}

impl From<PolynomialMapError> for RegressionError {
    fn from(value: PolynomialMapError) -> Self {
        Self::Map(value)
    }
}

/// Fit a multivariate polynomial map by least squares.
///
/// The basis contains every monomial u^alpha with |alpha| <= degree.
/// The implementation uses normal equations as a small dependency-free
/// baseline; a QR/SVD solver can replace this backend later.
pub fn fit_polynomial(
    samples: &[(Vec<f64>, Vec<f64>)],
    config: RegressionConfig,
) -> Result<PolynomialMap, RegressionError> {
    if samples.is_empty() {
        return Err(RegressionError::EmptySamples);
    }

    let parameter_dimension = samples[0].0.len();
    let output_dimension = samples[0].1.len();

    if parameter_dimension == 0 || output_dimension == 0 {
        return Err(RegressionError::NonFiniteSample);
    }

    if samples.iter().any(|(u, x)| {
        u.len() != parameter_dimension ||
        x.len() != output_dimension ||
        u.iter().any(|v| !v.is_finite()) ||
        x.iter().any(|v| !v.is_finite())
    }) {
        for (u, x) in samples {
            if u.len() != parameter_dimension {
                return Err(RegressionError::ParameterDimensionMismatch {
                    expected: parameter_dimension,
                    actual: u.len(),
                });
            }
            if x.len() != output_dimension {
                return Err(RegressionError::OutputDimensionMismatch {
                    expected: output_dimension,
                    actual: x.len(),
                });
            }
        }
        return Err(RegressionError::NonFiniteSample);
    }

    let basis = generate_multi_indices(parameter_dimension, config.degree);
    let term_count = basis.len();

    if samples.len() < term_count {
        return Err(RegressionError::Underdetermined {
            samples: samples.len(),
            terms: term_count,
        });
    }

    let mut gram = vec![vec![0.0; term_count]; term_count];
    let mut rhs = vec![vec![0.0; output_dimension]; term_count];

    for (u, x) in samples {
        let row: Vec<f64> = basis.iter().map(|index| evaluate_monomial(index, u)).collect();

        for i in 0..term_count {
            for j in 0..term_count {
                gram[i][j] += row[i] * row[j];
            }
            for q in 0..output_dimension {
                rhs[i][q] += row[i] * x[q];
            }
        }
    }

    let coefficients = solve_linear_system(gram, rhs)?;

    let mut map = PolynomialMap::new(parameter_dimension, output_dimension)?;
    for (index, coefficient) in basis.into_iter().zip(coefficients) {
        if coefficient.iter().any(|x| x.abs() > f64::EPSILON) {
            map.add_term(index, coefficient)?;
        }
    }

    Ok(map)
}

fn generate_multi_indices(dimension: usize, degree: usize) -> Vec<MultiIndex> {
    fn visit(
        position: usize,
        dimension: usize,
        remaining: usize,
        current: &mut Vec<usize>,
        output: &mut Vec<MultiIndex>,
    ) {
        if position == dimension {
            output.push(MultiIndex::new(current.clone()));
            return;
        }

        for exponent in 0..=remaining {
            current.push(exponent);
            visit(position + 1, dimension, remaining - exponent, current, output);
            current.pop();
        }
    }

    let mut output = Vec::new();
    visit(0, dimension, degree, &mut Vec::new(), &mut output);
    output
}

fn evaluate_monomial(index: &MultiIndex, point: &[f64]) -> f64 {
    index
        .exponents()
        .iter()
        .zip(point)
        .fold(1.0, |value, (&exponent, &x)| value * x.powi(exponent as i32))
}

fn solve_linear_system(
    mut matrix: Vec<Vec<f64>>,
    mut rhs: Vec<Vec<f64>>,
) -> Result<Vec<Vec<f64>>, RegressionError> {
    let n = matrix.len();
    let outputs = rhs[0].len();

    for pivot in 0..n {
        let mut best = pivot;
        for row in (pivot + 1)..n {
            if matrix[row][pivot].abs() > matrix[best][pivot].abs() {
                best = row;
            }
        }

        if matrix[best][pivot].abs() <= 1e-12 {
            return Err(RegressionError::SingularSystem);
        }

        matrix.swap(pivot, best);
        rhs.swap(pivot, best);

        let pivot_value = matrix[pivot][pivot];
        for column in pivot..n {
            matrix[pivot][column] /= pivot_value;
        }
        for q in 0..outputs {
            rhs[pivot][q] /= pivot_value;
        }

        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            if factor.abs() <= f64::EPSILON {
                continue;
            }
            for column in pivot..n {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            for q in 0..outputs {
                rhs[row][q] -= factor * rhs[pivot][q];
            }
        }
    }

    Ok(rhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_quadratic() {
        let samples = vec![
            (vec![-2.0], vec![9.0]),
            (vec![-1.0], vec![2.0]),
            (vec![0.0], vec![1.0]),
            (vec![1.0], vec![6.0]),
            (vec![2.0], vec![17.0]),
        ];

        let map = fit_polynomial(&samples, RegressionConfig::new(2)).unwrap();

        for (u, expected) in samples {
            let actual = map.evaluate(&u).unwrap()[0];
            assert!((actual - expected[0]).abs() < 1e-9);
        }
    }

    #[test]
    fn recovers_multivariate_map() {
        let samples = vec![
            (vec![0.0, 0.0], vec![0.0, 0.0]),
            (vec![1.0, 0.0], vec![1.0, 0.0]),
            (vec![0.0, 1.0], vec![1.0, 0.0]),
            (vec![1.0, 1.0], vec![2.0, 1.0]),
            (vec![2.0, 1.0], vec![3.0, 2.0]),
            (vec![1.0, 2.0], vec![3.0, 2.0]),
        ];

        let map = fit_polynomial(&samples, RegressionConfig::new(2)).unwrap();

        for (u, expected) in samples {
            let actual = map.evaluate(&u).unwrap();
            assert!((actual[0] - expected[0]).abs() < 1e-8);
            assert!((actual[1] - expected[1]).abs() < 1e-8);
        }
    }

    #[test]
    fn rejects_underdetermined_system() {
        let samples = vec![
            (vec![0.0], vec![1.0]),
            (vec![1.0], vec![2.0]),
        ];

        let error = fit_polynomial(&samples, RegressionConfig::new(2)).unwrap_err();

        assert!(matches!(error, RegressionError::Underdetermined { .. }));
    }
}
