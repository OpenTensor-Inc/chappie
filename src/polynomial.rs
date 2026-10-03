use std::fmt;

#[derive(Debug,Clone,PartialEq,Eq,Hash)]
pub struct MultiIndex { exponents:Vec<usize> }
impl MultiIndex {
    pub fn new(exponents:Vec<usize>)->Self{Self{exponents}}
    pub fn dimension(&self)->usize{self.exponents.len()}
    pub fn degree(&self)->usize{self.exponents.iter().sum()}
    pub fn exponents(&self)->&[usize]{&self.exponents}
}
#[derive(Debug,Clone,PartialEq)]
struct Term { index:MultiIndex, coefficient:Vec<f64> }
#[derive(Debug,Clone,PartialEq)]
pub struct PolynomialMap { parameter_dimension:usize, output_dimension:usize, terms:Vec<Term> }
impl PolynomialMap {
    pub fn new(parameter_dimension:usize,output_dimension:usize)->Result<Self,PolynomialMapError>{
        if parameter_dimension==0{return Err(PolynomialMapError::ZeroParameterDimension)}
        if output_dimension==0{return Err(PolynomialMapError::ZeroOutputDimension)}
        Ok(Self{parameter_dimension,output_dimension,terms:Vec::new()})
    }
    pub fn parameter_dimension(&self)->usize{self.parameter_dimension}
    pub fn output_dimension(&self)->usize{self.output_dimension}
    pub fn term_count(&self)->usize{self.terms.len()}
    pub fn degree(&self)->usize{self.terms.iter().map(|t|t.index.degree()).max().unwrap_or(0)}
    pub fn add_term(&mut self,index:MultiIndex,coefficient:Vec<f64>)->Result<(),PolynomialMapError>{
        if index.dimension()!=self.parameter_dimension{return Err(PolynomialMapError::InvalidMultiIndexDimension{expected:self.parameter_dimension,actual:index.dimension()})}
        if coefficient.len()!=self.output_dimension{return Err(PolynomialMapError::InvalidCoefficientDimension{expected:self.output_dimension,actual:coefficient.len()})}
        if coefficient.iter().any(|x|!x.is_finite()){return Err(PolynomialMapError::NonFiniteCoefficient)}
        if let Some(term)=self.terms.iter_mut().find(|t|t.index==index){for(a,b) in term.coefficient.iter_mut().zip(coefficient){*a+=b;}}else{self.terms.push(Term{index,coefficient});}
        Ok(())
    }
    pub fn evaluate(&self,point:&[f64])->Result<Vec<f64>,PolynomialMapError>{
        if point.len()!=self.parameter_dimension{return Err(PolynomialMapError::InvalidPointDimension{expected:self.parameter_dimension,actual:point.len()})}
        if point.iter().any(|x|!x.is_finite()){return Err(PolynomialMapError::NonFinitePoint)}
        let mut out=vec![0.0;self.output_dimension];
        for term in &self.terms{
            let monomial=term.index.exponents.iter().zip(point).fold(1.0,|a,(&p,&x)|a*x.powi(p as i32));
            for(y,c) in out.iter_mut().zip(&term.coefficient){*y+=c*monomial;}
        }
        if out.iter().any(|x|!x.is_finite()){return Err(PolynomialMapError::NonFiniteOutput)}
        Ok(out)
    }
    pub fn terms(&self)->impl Iterator<Item=(&MultiIndex,&[f64])>{self.terms.iter().map(|t|(&t.index,t.coefficient.as_slice()))}
}
#[derive(Debug,Clone,PartialEq)]
pub enum PolynomialMapError{ZeroParameterDimension,ZeroOutputDimension,InvalidMultiIndexDimension{expected:usize,actual:usize},InvalidCoefficientDimension{expected:usize,actual:usize},InvalidPointDimension{expected:usize,actual:usize},NonFiniteCoefficient,NonFinitePoint,NonFiniteOutput}
impl fmt::Display for PolynomialMapError{
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{
        Self::ZeroParameterDimension=>write!(f,"parameter dimension must be greater than zero"),
        Self::ZeroOutputDimension=>write!(f,"output dimension must be greater than zero"),
        Self::InvalidMultiIndexDimension{expected,actual}=>write!(f,"multi-index dimension {actual}, expected {expected}"),
        Self::InvalidCoefficientDimension{expected,actual}=>write!(f,"coefficient dimension {actual}, expected {expected}"),
        Self::InvalidPointDimension{expected,actual}=>write!(f,"point dimension {actual}, expected {expected}"),
        Self::NonFiniteCoefficient=>write!(f,"coefficients must be finite"),
        Self::NonFinitePoint=>write!(f,"evaluation point must be finite"),
        Self::NonFiniteOutput=>write!(f,"polynomial evaluation produced a non-finite value"),
    }}
}
impl std::error::Error for PolynomialMapError {}
#[cfg(test)]
mod tests{
 use super::*;
 #[test]fn polynomial(){let mut p=PolynomialMap::new(1,1).unwrap();p.add_term(MultiIndex::new(vec![0]),vec![1.]).unwrap();p.add_term(MultiIndex::new(vec![1]),vec![2.]).unwrap();p.add_term(MultiIndex::new(vec![2]),vec![3.]).unwrap();assert_eq!(p.evaluate(&[2.]).unwrap(),vec![17.]);}
 #[test]fn multivariate(){let mut p=PolynomialMap::new(2,2).unwrap();p.add_term(MultiIndex::new(vec![1,0]),vec![1.,0.]).unwrap();p.add_term(MultiIndex::new(vec![0,1]),vec![0.,1.]).unwrap();p.add_term(MultiIndex::new(vec![1,1]),vec![2.,3.]).unwrap();assert_eq!(p.evaluate(&[2.,4.]).unwrap(),vec![18.,52.]);}
}
