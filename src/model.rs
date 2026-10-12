//! Quality fields over tetrahedral meshes.
//!
//! A [`Field`] turns a point inside one of a mesh's simplices, given by its
//! barycentric coordinates, into a [`Quality`].  Each field type owns the
//! per-vertex or per-simplex data it reads, so a depth-profile field can't
//! exist without its profile; the mesh owns the connectivity and checks the
//! field covers it when the two are paired.  [`AnyField`] holds any of them
//! for meshes whose field type is only known at run time.

use crate::barycentric::interpolate;
use crate::quality::{Quality, barycentric_interpolate};
use crate::real::Real;
use enum_dispatch::enum_dispatch;
use nalgebra::Point4;

/// A 1-D velocity profile: piecewise-constant qualities as a function of
/// depth below a reference surface.
///
/// Layer `i` spans depths `[boundaries[i - 1], boundaries[i])`.  The first
/// layer extends upwards without limit and the last downwards without
/// limit, so every depth maps to a layer.  The `alpha` of each layer is
/// ignored: [`Profiled`] takes alpha from the mesh vertices.
#[derive(Debug, Clone, PartialEq)]
pub struct DepthProfile {
    boundaries: Vec<Real>,
    layers: Vec<Quality>,
}

/// Reasons a [`DepthProfile`] cannot be built.
#[derive(Debug, PartialEq)]
pub enum DepthProfileError {
    /// The profile has no layers.
    Empty,
    /// `tops` and `layers` differ in length.
    LengthMismatch,
    /// Layer tops are not strictly increasing (or are NaN).
    Unsorted,
}

impl std::fmt::Display for DepthProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Empty => "depth profile has no layers",
            Self::LengthMismatch => "profile tops and qualities differ in length",
            Self::Unsorted => "profile layer tops must strictly increase",
        })
    }
}

impl DepthProfile {
    /// Build a profile from the depth of the top of each layer and the
    /// layer qualities.  `tops[0]` is ignored, because the first layer
    /// extends upwards without limit.
    pub fn new(tops: &[Real], layers: Vec<Quality>) -> Result<Self, DepthProfileError> {
        if layers.is_empty() {
            return Err(DepthProfileError::Empty);
        }
        if tops.len() != layers.len() {
            return Err(DepthProfileError::LengthMismatch);
        }
        let boundaries = tops[1..].to_vec();
        if boundaries.iter().any(|b| b.is_nan()) || !boundaries.windows(2).all(|w| w[0] < w[1]) {
            return Err(DepthProfileError::Unsorted);
        }
        Ok(Self { boundaries, layers })
    }

    /// Return the quality of the layer containing `depth`.  A depth on a
    /// boundary belongs to the deeper layer.
    pub fn at(&self, depth: Real) -> Quality {
        self.layers[self.boundaries.partition_point(|&b| b <= depth)]
    }
}

/// A quality field over the simplices of a mesh.
///
/// The mesh owns the connectivity: it hands the field the simplex it found
/// and that simplex's four vertex indices.
#[enum_dispatch]
pub trait Field {
    /// Whether the field has data for every vertex and simplex of a mesh
    /// with `num_vertices` vertices and `num_simplices` simplices.
    fn covers(&self, num_vertices: usize, num_simplices: usize) -> bool;

    /// The quality at the point of simplex `simplex`, whose vertices are
    /// `corners`, with barycentric coordinates `bary` (in corner order).
    fn quality_at(&self, simplex: usize, corners: [u32; 4], bary: &Point4<Real>) -> Quality;

    /// Reorder per-simplex data so the new entry `i` is the old entry
    /// `order[i]`, to match the mesh's BVH leaf order.  Fields holding only per-vertex data
    /// have nothing to reorder.
    #[allow(unused_variables)]
    fn reorder(&mut self, order: &[u32]) {}

    /// Bytes the field holds on the heap.
    fn heap_size(&self) -> usize;
}

fn vec_bytes<T>(v: &Vec<T>) -> usize {
    v.capacity() * std::mem::size_of::<T>()
}

/// Barycentric interpolation of a quality per vertex.  Used by tomography.
#[derive(Debug, Clone, PartialEq)]
pub struct Interpolated {
    vertices: Vec<Quality>,
}

impl Interpolated {
    /// A field interpolating the quality of each vertex.
    pub fn new(vertices: Vec<Quality>) -> Self {
        Self { vertices }
    }
}

impl Field for Interpolated {
    fn covers(&self, num_vertices: usize, _num_simplices: usize) -> bool {
        self.vertices.len() == num_vertices
    }

    fn quality_at(&self, _simplex: usize, corners: [u32; 4], bary: &Point4<Real>) -> Quality {
        let corners = corners.map(|v| self.vertices[v as usize]);
        barycentric_interpolate(corners, [bary.x, bary.y, bary.z, bary.w])
    }

    fn heap_size(&self) -> usize {
        vec_bytes(&self.vertices)
    }
}

/// One quality per simplex.
#[derive(Debug, Clone, PartialEq)]
pub struct Constant {
    qualities: Vec<Quality>,
}

impl Constant {
    /// A field giving simplex `i` the quality `qualities[i]`.
    pub fn new(qualities: Vec<Quality>) -> Self {
        Self { qualities }
    }
}

impl Field for Constant {
    fn covers(&self, _num_vertices: usize, num_simplices: usize) -> bool {
        self.qualities.len() == num_simplices
    }

    fn quality_at(&self, simplex: usize, _corners: [u32; 4], _bary: &Point4<Real>) -> Quality {
        self.qualities[simplex]
    }

    fn reorder(&mut self, order: &[u32]) {
        self.qualities = order.iter().map(|&i| self.qualities[i as usize]).collect();
    }

    fn heap_size(&self) -> usize {
        vec_bytes(&self.qualities)
    }
}

/// A [`DepthProfile`] evaluated at the depth below a reference surface.  Used
/// by basins.
///
/// Depth and alpha are interpolated barycentrically from the vertices, so a
/// single simplex can span any number of profile layers: the layer
/// interfaces need no simplices of their own.
#[derive(Debug, Clone, PartialEq)]
pub struct Profiled {
    /// `[depth below the reference surface, alpha]` of each vertex, kept
    /// together so a query reads one array.
    vertices: Vec<[Real; 2]>,
    profile: DepthProfile,
}

impl Profiled {
    /// A field evaluating `profile` at the interpolated depth, with each
    /// vertex given as `[depth, alpha]`.
    pub fn new(vertices: Vec<[Real; 2]>, profile: DepthProfile) -> Self {
        Self { vertices, profile }
    }
}

impl Field for Profiled {
    fn covers(&self, num_vertices: usize, _num_simplices: usize) -> bool {
        self.vertices.len() == num_vertices
    }

    fn quality_at(&self, _simplex: usize, corners: [u32; 4], bary: &Point4<Real>) -> Quality {
        let corners = corners.map(|v| self.vertices[v as usize]);
        let weights = [bary.x, bary.y, bary.z, bary.w];
        let depth = interpolate(corners.map(|[d, _]| d), weights);
        let alpha = interpolate(corners.map(|[_, a]| a), weights);
        Quality {
            alpha,
            ..self.profile.at(depth)
        }
    }

    fn heap_size(&self) -> usize {
        vec_bytes(&self.vertices)
            + vec_bytes(&self.profile.boundaries)
            + vec_bytes(&self.profile.layers)
    }
}

/// Any of the field types, for meshes whose field type is only known at
/// run time.  Queries pay one dispatch per simplex hit.
#[enum_dispatch(Field)]
#[derive(Debug, Clone, PartialEq)]
pub enum AnyField {
    Interpolated,
    Constant,
    Profiled,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn layer(vs: Real) -> Quality {
        Quality {
            rho: 2000.0,
            vp: 2.0 * vs,
            vs,
            qp: 100.0,
            qs: 50.0,
            alpha: 1.0,
        }
    }

    fn three_layers() -> DepthProfile {
        DepthProfile::new(
            &[0.0, 100.0, 300.0],
            vec![layer(400.0), layer(800.0), layer(1600.0)],
        )
        .unwrap()
    }

    #[test]
    fn test_profile_picks_layer_by_depth() {
        let profile = three_layers();
        assert_eq!(profile.at(50.0).vs, 400.0);
        assert_eq!(profile.at(150.0).vs, 800.0);
        assert_eq!(profile.at(299.0).vs, 800.0);
        assert_eq!(profile.at(5000.0).vs, 1600.0);
    }

    #[test]
    fn test_profile_boundary_belongs_to_deeper_layer() {
        let profile = three_layers();
        assert_eq!(profile.at(100.0).vs, 800.0);
        assert_eq!(profile.at(300.0).vs, 1600.0);
    }

    #[test]
    fn test_profile_extends_above_first_top() {
        // Points above the reference surface (e.g. a padded top) take the
        // first layer.
        assert_eq!(three_layers().at(-50.0).vs, 400.0);
    }

    #[test]
    fn test_profile_rejects_bad_input() {
        assert_eq!(
            DepthProfile::new(&[], vec![]),
            Err(DepthProfileError::Empty)
        );
        assert_eq!(
            DepthProfile::new(&[0.0], vec![layer(1.0), layer(2.0)]),
            Err(DepthProfileError::LengthMismatch)
        );
        assert_eq!(
            DepthProfile::new(&[0.0, 200.0, 100.0], vec![layer(1.0); 3]),
            Err(DepthProfileError::Unsorted)
        );
        assert_eq!(
            DepthProfile::new(&[0.0, Real::NAN], vec![layer(1.0); 2]),
            Err(DepthProfileError::Unsorted)
        );
    }

    #[test]
    fn test_single_layer_profile_ignores_first_top() {
        let profile = DepthProfile::new(&[Real::NAN], vec![layer(700.0)]).unwrap();
        assert_eq!(profile.at(-1e6).vs, 700.0);
        assert_eq!(profile.at(1e6).vs, 700.0);
    }

    /// The barycentric coordinates of vertex `i` of a simplex.
    fn at_vertex(i: usize) -> Point4<Real> {
        let mut bary = Point4::origin();
        bary[i] = 1.0;
        bary
    }

    #[test]
    fn test_interpolated_reproduces_vertices() {
        let field = Interpolated::new((0..5).map(|i| layer(i as Real)).collect());
        let corners = [4, 2, 0, 1];
        for (k, v) in corners.into_iter().enumerate() {
            assert_eq!(field.quality_at(0, corners, &at_vertex(k)).vs, v as Real);
        }
        let centre = Point4::new(0.25, 0.25, 0.25, 0.25);
        assert_relative_eq!(field.quality_at(0, corners, &centre).vs, 7.0 / 4.0);
    }

    /// One simplex spanning all three layers: the profile is evaluated at
    /// the interpolated depth, and alpha is interpolated from the vertices.
    #[test]
    fn test_profiled_spans_layers() {
        // Vertices 0-2 on the reference surface, vertex 3 400 m below it.
        let vertices = vec![[0.0, 1.0], [0.0, 1.0], [0.0, 0.5], [400.0, 0.5]];
        let field = Profiled::new(vertices, three_layers());
        let corners = [0, 1, 2, 3];
        for (depth, vs) in [(50.0, 400.0), (150.0, 800.0), (350.0, 1600.0)] {
            let w = depth / 400.0;
            let bary = Point4::new((1.0 - w) / 3.0, (1.0 - w) / 3.0, (1.0 - w) / 3.0, w);
            assert_eq!(field.quality_at(0, corners, &bary).vs, vs, "depth {depth}");
        }
        let bary = Point4::new(0.4, 0.3, 0.2, 0.1);
        assert_relative_eq!(
            field.quality_at(0, corners, &bary).alpha,
            0.4 + 0.3 + 0.5 * 0.3
        );
    }

    #[test]
    fn test_constant_ignores_position() {
        let field = Constant::new(vec![layer(1.0), layer(2.0)]);
        assert_eq!(field.quality_at(1, [0; 4], &at_vertex(0)).vs, 2.0);
        assert_eq!(field.quality_at(1, [0; 4], &at_vertex(3)).vs, 2.0);
    }

    #[test]
    fn test_coverage() {
        assert!(Interpolated::new(vec![layer(0.0); 4]).covers(4, 9));
        assert!(!Interpolated::new(vec![layer(0.0); 3]).covers(4, 9));
        assert!(Profiled::new(vec![[0.0, 1.0]; 4], three_layers()).covers(4, 9));
        assert!(Constant::new(vec![layer(0.0); 9]).covers(4, 9));
        assert!(!Constant::new(vec![layer(0.0); 4]).covers(4, 9));
    }

    #[test]
    fn test_reorder_permutes_simplices() {
        let mut field = AnyField::from(Constant::new(vec![layer(0.0), layer(1.0), layer(2.0)]));
        field.reorder(&[2, 0, 1]);
        let vs: Vec<Real> = (0..3)
            .map(|i| field.quality_at(i, [0; 4], &at_vertex(0)).vs)
            .collect();
        assert_eq!(vs, [2.0, 0.0, 1.0]);
    }
}
