use std::fmt;

/// A point in an arbitrary-dimensional Euclidean space.
///
/// `Point<d>` is represented dynamically as a vector so that
/// the first implementation can support arbitrary dimensions.
pub type Point = Vec<f64>;

/// A rectangular domain in parameter space.
///
/// For a d-dimensional chart:
///
///     U = [min_0, max_0] × ... × [min_d, max_d]
///
/// The domain is closed and bounded.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    bounds: Vec<(f64, f64)>,
}

impl Domain {
    pub fn new(bounds: Vec<(f64, f64)>) -> Result<Self, DomainError> {
        if bounds.is_empty() {
            return Err(DomainError::Empty);
        }

        for &(min, max) in &bounds {
            if !min.is_finite() || !max.is_finite() {
                return Err(DomainError::NonFinite);
            }

            if min > max {
                return Err(DomainError::InvalidBounds { min, max });
            }
        }

        Ok(Self { bounds })
    }

    pub fn dimension(&self) -> usize {
        self.bounds.len()
    }

    pub fn contains(&self, point: &[f64]) -> bool {
        if point.len() != self.dimension() {
            return false;
        }

        point
            .iter()
            .zip(self.bounds.iter())
            .all(|(&x, &(min, max))| x >= min && x <= max)
    }

    pub fn bounds(&self) -> &[(f64, f64)] {
        &self.bounds
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    Empty,
    NonFinite,
    InvalidBounds { min: f64, max: f64 },
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "domain must contain at least one dimension"),
            Self::NonFinite => write!(f, "domain bounds must be finite"),
            Self::InvalidBounds { min, max } => {
                write!(f, "invalid bounds: {min} > {max}")
            }
        }
    }
}

impl std::error::Error for DomainError {}

/// A coordinate chart.
///
/// Mathematically:
///
///     φ : U ⊂ R^d → R^n
///
/// The function itself is stored as a Rust closure.
///
/// Later, this will be replaced/augmented by a compact mathematical
/// approximation.
pub struct Chart {
    domain: Domain,
    embedding_dimension: usize,
    map: Box<dyn Fn(&[f64]) -> Point + Send + Sync>,
}

impl Chart {
    pub fn new<F>(
        domain: Domain,
        embedding_dimension: usize,
        map: F,
    ) -> Result<Self, ChartError>
    where
        F: Fn(&[f64]) -> Point + Send + Sync + 'static,
    {
        if embedding_dimension == 0 {
            return Err(ChartError::ZeroEmbeddingDimension);
        }

        Ok(Self {
            domain,
            embedding_dimension,
            map: Box::new(map),
        })
    }

    pub fn parameter_dimension(&self) -> usize {
        self.domain.dimension()
    }

    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    pub fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Evaluate φ(x).
    pub fn evaluate(&self, parameter: &[f64]) -> Result<Point, ChartError> {
        if !self.domain.contains(parameter) {
            return Err(ChartError::OutsideDomain);
        }

        let point = (self.map)(parameter);

        if point.len() != self.embedding_dimension {
            return Err(ChartError::InvalidMapDimension {
                expected: self.embedding_dimension,
                actual: point.len(),
            });
        }

        if point.iter().any(|x| !x.is_finite()) {
            return Err(ChartError::NonFiniteOutput);
        }

        Ok(point)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChartError {
    ZeroEmbeddingDimension,
    OutsideDomain,
    InvalidMapDimension { expected: usize, actual: usize },
    NonFiniteOutput,
}

impl fmt::Display for ChartError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEmbeddingDimension => {
                write!(f, "embedding dimension must be greater than zero")
            }
            Self::OutsideDomain => {
                write!(f, "parameter lies outside chart domain")
            }
            Self::InvalidMapDimension { expected, actual } => {
                write!(
                    f,
                    "chart returned dimension {actual}, expected {expected}"
                )
            }
            Self::NonFiniteOutput => {
                write!(f, "chart returned a non-finite coordinate")
            }
        }
    }
}

impl std::error::Error for ChartError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_contains_points() {
        let domain = Domain::new(vec![
            (-1.0, 1.0),
            (0.0, 2.0),
        ])
        .unwrap();

        assert!(domain.contains(&[0.0, 1.0]));
        assert!(domain.contains(&[-1.0, 0.0]));
        assert!(domain.contains(&[1.0, 2.0]));

        assert!(!domain.contains(&[2.0, 1.0]));
    }

    #[test]
    fn chart_evaluates_function() {
        let domain = Domain::new(vec![
            (-1.0, 1.0),
            (-1.0, 1.0),
        ])
        .unwrap();

        let chart = Chart::new(domain, 3, |p| {
            vec![
                p[0],
                p[1],
                p[0] * p[1],
            ]
        })
        .unwrap();

        let result = chart.evaluate(&[2.0, 3.0]);

        assert!(matches!(result, Err(ChartError::OutsideDomain)));

        let result = chart.evaluate(&[0.5, 0.25]).unwrap();

        assert_eq!(result, vec![0.5, 0.25, 0.125]);
    }

    #[test]
    fn rejects_wrong_output_dimension() {
        let domain = Domain::new(vec![(-1.0, 1.0)]).unwrap();

        let chart = Chart::new(domain, 3, |_p| {
            vec![1.0, 2.0]
        })
        .unwrap();

        let result = chart.evaluate(&[0.0]);

        assert!(matches!(
            result,
            Err(ChartError::InvalidMapDimension {
                expected: 3,
                actual: 2
            })
        ));
    }

    #[test]
    fn rejects_non_finite_output() {
        let domain = Domain::new(vec![(-1.0, 1.0)]).unwrap();

        let chart = Chart::new(domain, 1, |_p| {
            vec![f64::NAN]
        })
        .unwrap();

        let result = chart.evaluate(&[0.0]);

        assert!(matches!(result, Err(ChartError::NonFiniteOutput)));
    }
}