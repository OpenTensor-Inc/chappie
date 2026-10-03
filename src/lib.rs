pub mod atlas;
pub mod chart;
pub mod domain;
pub mod error;
pub mod polynomial;
pub mod regression;
pub mod topology;

pub use atlas::{Atlas, AtlasError, ChartId, TransitionMap};
pub use chart::{Chart, ChartError};
pub use domain::{Domain, DomainError, Point};
pub use error::{coverage_radius, polynomial_lipschitz_bound, CoverageErrorBound, ErrorBoundError, SampleError};
pub use polynomial::{MultiIndex, PolynomialMap, PolynomialMapError};
pub use regression::{fit_polynomial, RegressionConfig, RegressionError};
pub use topology::{Edge, Topology, TopologyError, Triangle, VertexId};
