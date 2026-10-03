use crate::{Domain, Point, PolynomialMap};
use std::fmt;

#[derive(Debug,Clone,PartialEq)]
pub struct Chart { domain:Domain, map:PolynomialMap }
impl Chart {
 pub fn new(domain:Domain,map:PolynomialMap)->Result<Self,ChartError>{
  if domain.dimension()!=map.parameter_dimension(){return Err(ChartError::ParameterDimensionMismatch{domain:domain.dimension(),map:map.parameter_dimension()})}
  Ok(Self{domain,map})
 }
 pub fn parameter_dimension(&self)->usize{self.domain.dimension()}
 pub fn embedding_dimension(&self)->usize{self.map.output_dimension()}
 pub fn domain(&self)->&Domain{&self.domain}
 pub fn map(&self)->&PolynomialMap{&self.map}
 pub fn evaluate(&self,parameter:&[f64])->Result<Point,ChartError>{
  if !self.domain.contains(parameter){return Err(ChartError::OutsideDomain)}
  self.map.evaluate(parameter).map_err(ChartError::Map)
 }
}
#[derive(Debug,Clone,PartialEq)]
pub enum ChartError{ParameterDimensionMismatch{domain:usize,map:usize},OutsideDomain,Map(crate::PolynomialMapError)}
impl fmt::Display for ChartError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{
 Self::ParameterDimensionMismatch{domain,map}=>write!(f,"domain dimension {domain} does not match map dimension {map}"),
 Self::OutsideDomain=>write!(f,"parameter lies outside chart domain"),Self::Map(e)=>write!(f,"chart map error: {e}")}}}
impl std::error::Error for ChartError {}
