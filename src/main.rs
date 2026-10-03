use chappie::{Chart, Domain, MultiIndex, PolynomialMap};

fn main() {
    let domain = Domain::new(vec![(-1.0, 1.0), (-1.0, 1.0)]).expect("valid domain");
    let mut map = PolynomialMap::new(2, 3).expect("valid map");
    map.add_term(MultiIndex::new(vec![1, 0]), vec![1.0, 0.0, 0.0]).unwrap();
    map.add_term(MultiIndex::new(vec![0, 1]), vec![0.0, 1.0, 0.0]).unwrap();
    map.add_term(MultiIndex::new(vec![1, 1]), vec![0.0, 0.0, 1.0]).unwrap();
    let chart = Chart::new(domain, map).expect("valid chart");
    println!("{:?}", chart.evaluate(&[0.5, 0.25]).expect("valid point"));
}
