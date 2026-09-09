use crate::real::Real;
use crate::tree_query::contains_point_iterator; // Assumed 2D iterator
use crate::triangle::Triangle; // Assumed 2D equivalent of Simplex
use deepsize::{Context, DeepSizeOf};

use bvh::{
    bounding_hierarchy::BoundingHierarchy,
    bvh::{Bvh, BvhNode},
};
use nalgebra::{Point2, Point3};
use serde::Serialize;

/// Serialisable summary of a [`SurfaceModel`] for diagnostics.
#[derive(Serialize)]
pub struct SurfaceModelView {
    pub size: usize,
}

/// A 2D triangular surface model with an internal 2D BVH for fast elevation queries.
pub struct SurfaceModel {
    /// Internal BVH built in 2D space (X, Y)
    bvh_tree: Bvh<Real, 2>,
    triangles: Vec<Triangle>,
    model_map: Vec<Point3<usize>>,
    z: Vec<Real>,
}

impl DeepSizeOf for SurfaceModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.triangles.deep_size_of_children(context)
            + self.z.deep_size_of_children(context)
            + self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 2>>()
    }
}

impl SurfaceModel {
    /// Create a surface model from raw geometry.
    ///
    pub fn new(
        vertices: Vec<Point2<Real>>,
        faces: Vec<Point3<usize>>, // Triangle indices
        z: Vec<Real>,
    ) -> Self {
        // Build 2D Triangles for the internal BVH.  Degenerate (collinear in
        // XY) faces have no barycentric basis and are skipped; `Triangle::id`
        // still indexes into `faces`/`model_map`, so dropping them is safe.
        let mut triangles: Vec<Triangle> = faces
            .iter()
            .enumerate()
            .filter_map(|(i, f)| Triangle::new(vertices[f.x], vertices[f.y], vertices[f.z], i))
            .collect();

        // BVH is built in 2D space (Triangles must implement Bounded<Real, 2>)
        let bvh_tree = Bvh::build_par(&mut triangles);

        Self {
            bvh_tree,
            triangles,
            z,
            model_map: faces,
        }
    }

    /// Interpolate quality at (x, y) by finding the triangle and using barycentric coordinates.
    pub fn query(&self, point: Point2<Real>) -> Option<Real> {
        contains_point_iterator(&self.bvh_tree, &self.triangles, &point)
            .next()
            .map(|tri| {
                let bary = tri.barycentric_coordinates(&point);
                let model = self.model_map[tri.id];
                let z0 = self.z[model.x];
                let z1 = self.z[model.y];
                let z2 = self.z[model.z];
                bary.x * z0 + bary.y * z1 + bary.z * z2
            })
    }

    pub fn view(&self) -> SurfaceModelView {
        SurfaceModelView {
            size: self.deep_size_of(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// The linear field the test surfaces sample: `z = 3x - 2y + 5`.
    ///
    /// Barycentric interpolation reproduces a linear field *exactly*, which
    /// gives every test below a free analytic oracle — one that catches wrong
    /// vertex ordering, wrong weights and wrong triangle selection at once.
    fn plane(x: Real, y: Real) -> Real {
        3.0 * x - 2.0 * y + 5.0
    }

    /// An `n x n` grid over `[0, n-1]^2`, split into two triangles per cell.
    fn grid_surface(n: usize) -> SurfaceModel {
        let mut vertices = Vec::new();
        let mut z = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (x, y) = (i as Real, j as Real);
                vertices.push(Point2::new(x, y));
                z.push(plane(x, y));
            }
        }

        let idx = |i: usize, j: usize| j * n + i;
        let mut faces = Vec::new();
        for j in 0..n - 1 {
            for i in 0..n - 1 {
                faces.push(Point3::new(idx(i, j), idx(i + 1, j), idx(i, j + 1)));
                faces.push(Point3::new(idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)));
            }
        }

        SurfaceModel::new(vertices, faces, z)
    }

    #[test]
    fn test_query_reproduces_linear_field_at_vertices() {
        let s = grid_surface(4);
        for j in 0..4 {
            for i in 0..4 {
                let (x, y) = (i as Real, j as Real);
                let got = s.query(Point2::new(x, y)).expect("vertex must be covered");
                assert!(
                    (got - plane(x, y)).abs() < 1e-4,
                    "at ({x}, {y}): {got} != {}",
                    plane(x, y)
                );
            }
        }
    }

    #[test]
    fn test_query_reproduces_linear_field_in_cell_interiors() {
        let s = grid_surface(4);
        // Deliberately off-centre and off-diagonal so the two triangles of a
        // cell are distinguishable and the weights are all distinct.
        for (x, y) in [(0.3, 0.2), (0.7, 0.9), (1.25, 2.6), (2.5, 0.1), (1.1, 1.1)] {
            let got = s
                .query(Point2::new(x, y))
                .expect("interior must be covered");
            assert!(
                (got - plane(x, y)).abs() < 1e-4,
                "at ({x}, {y}): {got} != {}",
                plane(x, y)
            );
        }
    }

    #[test]
    fn test_query_outside_returns_none() {
        let s = grid_surface(3);
        assert!(s.query(Point2::new(-1.0, -1.0)).is_none());
        assert!(s.query(Point2::new(10.0, 0.5)).is_none());
        assert!(s.query(Point2::new(0.5, 10.0)).is_none());
    }

    /// Degenerate faces must be skipped rather than panic the constructor, and
    /// the surviving triangles must still index the right vertices.
    ///
    /// This is the regression test for `Triangle::new` returning `Option`: the
    /// collinear face is dropped, but `Triangle::id` still indexes `faces`, so
    /// the good face must interpolate correctly.
    #[test]
    fn test_degenerate_faces_are_skipped_not_fatal() {
        let vertices = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0), // collinear with the first two
            Point2::new(0.0, 4.0),
            Point2::new(4.0, 0.0),
        ];
        let z: Vec<Real> = vertices.iter().map(|p| plane(p.x, p.y)).collect();
        let faces = vec![
            Point3::new(0usize, 1, 2), // degenerate -> dropped
            Point3::new(0usize, 3, 4), // fine
        ];

        let s = SurfaceModel::new(vertices, faces, z);
        let p = Point2::new(1.0, 1.0);
        let got = s
            .query(p)
            .expect("the non-degenerate face must still resolve");
        assert!((got - plane(1.0, 1.0)).abs() < 1e-4, "{got}");
    }

    proptest! {
        /// Every point inside the surface's footprint interpolates to the exact
        /// value of the linear field.
        #[test]
        fn prop_query_matches_linear_field(
            x in (0.0 as Real)..3.0,
            y in (0.0 as Real)..3.0,
        ) {
            let s = grid_surface(4);
            let got = s.query(Point2::new(x, y)).expect("inside footprint");
            let want = plane(x, y);
            prop_assert!((got - want).abs() < 1e-3, "at ({x}, {y}): {got} != {want}");
        }
    }
}
