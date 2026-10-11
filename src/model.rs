//! Quality fields over tetrahedral meshes.
//!
//! A [`Field`] turns a point inside one of a mesh's simplices, given by its
//! barycentric coordinates, into a [`Quality`].  Each field type owns the
//! per-vertex or per-simplex data it reads; the mesh owns the connectivity
//! and checks the field covers it when the two are paired.  [`AnyField`] holds any of them
//! for meshes whose field type is only known at run time.

use crate::quality::{Quality, barycentric_interpolate};
use crate::real::Real;
use enum_dispatch::enum_dispatch;
use nalgebra::Point4;

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

/// Any of the field types, for meshes whose field type is only known at
/// run time.  Queries pay one dispatch per simplex hit.
#[enum_dispatch(Field)]
#[derive(Debug, Clone, PartialEq)]
pub enum AnyField {
    Interpolated,
    Constant,
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
