use crate::Chart;
use std::fmt;

#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)] pub struct ChartId(pub usize);
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)] pub struct TransitionMap{pub from:ChartId,pub to:ChartId}
pub struct Atlas{charts:Vec<Chart>,transitions:Vec<TransitionMap>}
impl Atlas{
 pub fn new()->Self{Self{charts:Vec::new(),transitions:Vec::new()}}
 pub fn add_chart(&mut self,chart:Chart)->ChartId{let id=ChartId(self.charts.len());self.charts.push(chart);id}
 pub fn add_transition(&mut self,from:ChartId,to:ChartId)->Result<(),AtlasError>{self.require_chart(from)?;self.require_chart(to)?;if from==to{return Err(AtlasError::SelfTransition)}if self.transitions.iter().any(|t|t.from==from&&t.to==to){return Err(AtlasError::DuplicateTransition{from,to})}self.transitions.push(TransitionMap{from,to});Ok(())}
 pub fn chart(&self,id:ChartId)->Result<&Chart,AtlasError>{self.charts.get(id.0).ok_or(AtlasError::UnknownChart(id))}
 pub fn transition(&self,from:ChartId,to:ChartId)->Result<&TransitionMap,AtlasError>{self.transitions.iter().find(|t|t.from==from&&t.to==to).ok_or(AtlasError::MissingTransition{from,to})}
 pub fn chart_count(&self)->usize{self.charts.len()} pub fn transition_count(&self)->usize{self.transitions.len()}
 fn require_chart(&self,id:ChartId)->Result<(),AtlasError>{if self.charts.get(id.0).is_some(){Ok(())}else{Err(AtlasError::UnknownChart(id))}}
}
impl Default for Atlas{fn default()->Self{Self::new()}}
#[derive(Debug,Clone,PartialEq,Eq)] pub enum AtlasError{UnknownChart(ChartId),MissingTransition{from:ChartId,to:ChartId},SelfTransition,DuplicateTransition{from:ChartId,to:ChartId}}
impl fmt::Display for AtlasError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{match self{Self::UnknownChart(id)=>write!(f,"unknown chart {}",id.0),Self::MissingTransition{from,to}=>write!(f,"missing transition from {} to {}",from.0,to.0),Self::SelfTransition=>write!(f,"a chart cannot transition to itself"),Self::DuplicateTransition{from,to}=>write!(f,"transition {} -> {} already exists",from.0,to.0)}}}
impl std::error::Error for AtlasError{}
