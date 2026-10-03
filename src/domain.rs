use std::fmt;

pub type Point = Vec<f64>;

#[derive(Debug, Clone, PartialEq)]
pub struct Domain { bounds: Vec<(f64, f64)> }

impl Domain {
    pub fn new(bounds: Vec<(f64, f64)>) -> Result<Self, DomainError> {
        if bounds.is_empty() { return Err(DomainError::Empty); }
        for &(min, max) in &bounds {
            if !min.is_finite() || !max.is_finite() { return Err(DomainError::NonFinite); }
            if min > max { return Err(DomainError::InvalidBounds { min, max }); }
        }
        Ok(Self { bounds })
    }
    pub fn dimension(&self) -> usize { self.bounds.len() }
    pub fn bounds(&self) -> &[(f64, f64)] { &self.bounds }
    pub fn contains(&self, point: &[f64]) -> bool {
        point.len() == self.dimension() &&
            point.iter().zip(&self.bounds).all(|(&x, &(min,max))| x.is_finite() && x >= min && x <= max)
    }
}
#[derive(Debug,Clone,PartialEq)]
pub enum DomainError { Empty, NonFinite, InvalidBounds { min:f64, max:f64 } }
impl fmt::Display for DomainError {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        match self { Self::Empty=>write!(f,"domain must contain at least one dimension"), Self::NonFinite=>write!(f,"domain bounds must be finite"), Self::InvalidBounds{min,max}=>write!(f,"invalid bounds: {min} > {max}") }
    }
}
impl std::error::Error for DomainError {}
