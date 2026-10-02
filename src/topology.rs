use std::collections::HashSet;
use std::fmt;

/// A vertex in the abstract topology.
///
/// A vertex has no geometric position yet.
/// Its position will belong to the geometry layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VertexId(pub usize);

/// A directed edge.
///
/// For topology we normalize an edge so that:
/// Edge(a, b) == Edge(b, a).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Edge {
    pub a: VertexId,
    pub b: VertexId,
}

impl Edge {
    pub fn new(a: VertexId, b: VertexId) -> Self {
        assert_ne!(a, b, "an edge cannot connect a vertex to itself");

        if a.0 < b.0 {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

/// A triangle.
///
/// The ordering is canonicalized so that the same triangle
/// cannot accidentally be inserted multiple times with different
/// vertex ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Triangle {
    pub vertices: [VertexId; 3],
}

impl Triangle {
    pub fn new(a: VertexId, b: VertexId, c: VertexId) -> Self {
        assert!(
            a != b && a != c && b != c,
            "a triangle needs three distinct vertices"
        );

        let mut vertices = [a, b, c];

        vertices.sort_by_key(|vertex| vertex.0);

        Self { vertices }
    }

    pub fn edges(&self) -> [Edge; 3] {
        [
            Edge::new(self.vertices[0], self.vertices[1]),
            Edge::new(self.vertices[1], self.vertices[2]),
            Edge::new(self.vertices[0], self.vertices[2]),
        ]
    }
}

/// The combinatorial topology of a triangulated space.
///
/// This structure intentionally contains no coordinates.
#[derive(Debug, Default, Clone)]
pub struct Topology {
    vertices: HashSet<VertexId>,
    edges: HashSet<Edge>,
    triangles: HashSet<Triangle>,
}

impl Topology {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_vertex(&mut self) -> VertexId {
        let id = VertexId(self.vertices.len());
        self.vertices.insert(id);
        id
    }

    pub fn add_triangle(
        &mut self,
        a: VertexId,
        b: VertexId,
        c: VertexId,
    ) -> Result<(), TopologyError> {
        self.require_vertex(a)?;
        self.require_vertex(b)?;
        self.require_vertex(c)?;

        let triangle = Triangle::new(a, b, c);

        if self.triangles.contains(&triangle) {
            return Err(TopologyError::DuplicateTriangle(triangle));
        }

        for edge in triangle.edges() {
            self.edges.insert(edge);
        }

        self.triangles.insert(triangle);

        Ok(())
    }

    fn require_vertex(&self, vertex: VertexId) -> Result<(), TopologyError> {
        if self.vertices.contains(&vertex) {
            Ok(())
        } else {
            Err(TopologyError::UnknownVertex(vertex))
        }
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    pub fn vertices(&self) -> impl Iterator<Item = VertexId> + '_ {
        self.vertices.iter().copied()
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.edges.iter().copied()
    }

    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.triangles.iter().copied()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TopologyError {
    UnknownVertex(VertexId),
    DuplicateTriangle(Triangle),
}

impl fmt::Display for TopologyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownVertex(vertex) => {
                write!(f, "unknown vertex {}", vertex.0)
            }
            Self::DuplicateTriangle(triangle) => {
                write!(
                    f,
                    "triangle ({}, {}, {}) already exists",
                    triangle.vertices[0].0, triangle.vertices[1].0, triangle.vertices[2].0
                )
            }
        }
    }
}

impl std::error::Error for TopologyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_vertices() {
        let mut topology = Topology::new();

        let v0 = topology.add_vertex();
        let v1 = topology.add_vertex();
        let v2 = topology.add_vertex();

        assert_eq!(v0, VertexId(0));
        assert_eq!(v1, VertexId(1));
        assert_eq!(v2, VertexId(2));

        assert_eq!(topology.vertex_count(), 3);
    }

    #[test]
    fn triangle_creates_edges() {
        let mut topology = Topology::new();

        let v0 = topology.add_vertex();
        let v1 = topology.add_vertex();
        let v2 = topology.add_vertex();

        topology
            .add_triangle(v0, v1, v2)
            .expect("triangle should be valid");

        assert_eq!(topology.vertex_count(), 3);
        assert_eq!(topology.edge_count(), 3);
        assert_eq!(topology.triangle_count(), 1);
    }

    #[test]
    fn adjacent_triangles_share_edge() {
        let mut topology = Topology::new();

        let v0 = topology.add_vertex();
        let v1 = topology.add_vertex();
        let v2 = topology.add_vertex();
        let v3 = topology.add_vertex();

        topology
            .add_triangle(v0, v1, v2)
            .expect("first triangle should be valid");

        topology
            .add_triangle(v1, v3, v2)
            .expect("second triangle should be valid");

        assert_eq!(topology.triangle_count(), 2);

        // Two triangles sharing one edge have five unique edges.
        assert_eq!(topology.edge_count(), 5);
    }

    #[test]
    fn triangle_order_is_canonical() {
        let a = VertexId(4);
        let b = VertexId(1);
        let c = VertexId(3);

        let t1 = Triangle::new(a, b, c);
        let t2 = Triangle::new(c, a, b);

        assert_eq!(t1, t2);
    }

    #[test]
    fn duplicate_triangle_is_rejected() {
        let mut topology = Topology::new();

        let v0 = topology.add_vertex();
        let v1 = topology.add_vertex();
        let v2 = topology.add_vertex();

        topology
            .add_triangle(v0, v1, v2)
            .expect("first triangle should be valid");

        let result = topology.add_triangle(v2, v0, v1);

        assert!(matches!(result, Err(TopologyError::DuplicateTriangle(_))));
    }

    #[test]
    fn unknown_vertex_is_rejected() {
        let mut topology = Topology::new();

        let v0 = topology.add_vertex();
        let v1 = topology.add_vertex();
        let unknown = VertexId(99);

        let result = topology.add_triangle(v0, v1, unknown);

        assert!(matches!(
            result,
            Err(TopologyError::UnknownVertex(VertexId(99)))
        ));
    }
}
