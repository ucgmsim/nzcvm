use crate::real::Real;
use crate::tree_query::Contains;
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use deepsize::{Context, DeepSizeOf};

use nalgebra::{Matrix2, Point2, Point3, Vector2};

const CONTAINMENT_EPS: Real = 1e-4;

/// A 2D triangle with pre-computed 2D inverse matrix for fast $x,y$ containment queries.
///
/// Vertex `c2` is the "anchor"; $c_0$ and $c_1$ are stored implicitly through the
/// inverse of the $2 \times 2$ matrix $[(c_0-c_2)_{xy}, (c_1-c_2)_{xy}]$.
#[derive(Clone, Copy, Debug)]
pub struct Triangle {
    pub c2: Point2<Real>,
    /// Precomputed inverse of the 2D basis mapping $l_0, l_1$ to $x, y$.
    inv_matrix: Matrix2<Real>,

    /// 2D AABB for internal surface BVH queries.
    aabb: Aabb<Real, 2>,

    pub id: usize,
    node_index: usize,
}

impl DeepSizeOf for Triangle {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

impl Triangle {
    /// Construct a new triangle from three vertices.
    ///
    /// Note: The BVH and containment logic use the XY projection.
    /// The Z-coordinates are preserved for interpolation.
    ///
    /// Returns `None` if the three vertices are collinear in the XY plane
    /// (zero area), mirroring [`Simplex::new`](crate::simplex::Simplex::new).
    /// Real surface meshes do contain occasional degenerate triangles; those
    /// are skipped by [`SurfaceModel::new`](crate::surface::SurfaceModel::new)
    /// rather than taking the process down.
    pub fn new(c0: Point2<Real>, c1: Point2<Real>, c2: Point2<Real>, id: usize) -> Option<Self> {
        let pts = [c0, c1, c2];

        // 2D AABB construction (XY plane)
        let min_p = Point2::new(
            pts.iter().map(|p| p.x).fold(Real::MAX, Real::min),
            pts.iter().map(|p| p.y).fold(Real::MAX, Real::min),
        );
        let max_p = Point2::new(
            pts.iter().map(|p| p.x).fold(Real::MIN, Real::max),
            pts.iter().map(|p| p.y).fold(Real::MIN, Real::max),
        );
        let aabb = Aabb::with_bounds(min_p, max_p);

        // Basis vectors in 2D (XY plane)
        let v0 = Vector2::new(c0.x - c2.x, c0.y - c2.y);
        let v1 = Vector2::new(c1.x - c2.x, c1.y - c2.y);

        let m = Matrix2::from_columns(&[v0, v1]);
        let inv_matrix = m.try_inverse()?;

        Some(Self {
            c2,
            inv_matrix,
            aabb,
            id,
            node_index: 0,
        })
    }

    /// Return the 3-element barycentric coordinates $(l_0, l_1, l_2)$ for point `p`.
    ///
    /// Used for interpolating elevation ($z$) or other qualities.
    #[inline(always)]
    pub fn barycentric_coordinates(&self, p: &Point2<Real>) -> Point3<Real> {
        let diff = Vector2::new(p.x - self.c2.x, p.y - self.c2.y);
        let l = self.inv_matrix * diff;

        let l0 = l.x;
        let l1 = l.y;
        let l2 = 1.0 - l0 - l1;

        Point3::new(l0, l1, l2)
    }
}

/// Contains implementation for 2D space.
impl Contains<Real, 2, Triangle> for Triangle {
    #[inline(always)]
    fn contains(&self, query_point: &Point2<Real>) -> Option<Triangle> {
        let diff = Vector2::new(query_point.x - self.c2.x, query_point.y - self.c2.y);

        let l0 = self.inv_matrix[(0, 0)] * diff.x + self.inv_matrix[(0, 1)] * diff.y;
        let l1 = self.inv_matrix[(1, 0)] * diff.x + self.inv_matrix[(1, 1)] * diff.y;

        if (l0 >= -CONTAINMENT_EPS)
            && (l1 >= -CONTAINMENT_EPS)
            && (l0 + l1 <= 1.0 + CONTAINMENT_EPS)
        {
            Some(*self)
        } else {
            None
        }
    }
}

impl Bounded<Real, 2> for Triangle {
    fn aabb(&self) -> Aabb<Real, 2> {
        self.aabb
    }
}

impl BHShape<Real, 2> for Triangle {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }
    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Twice the signed area of the XY triangle — the determinant `new` inverts.
    fn area2(c0: Point2<Real>, c1: Point2<Real>, c2: Point2<Real>) -> Real {
        ((c0.x - c2.x) * (c1.y - c2.y) - (c1.x - c2.x) * (c0.y - c2.y)).abs()
    }

    fn coord() -> impl Strategy<Value = Real> {
        (-50.0 as Real)..(50.0 as Real)
    }

    fn vertex() -> impl Strategy<Value = Point2<Real>> {
        (coord(), coord()).prop_map(|(x, y)| Point2::new(x, y))
    }

    /// Three non-negative weights summing to 1.
    fn weights() -> impl Strategy<Value = [Real; 3]> {
        [(0.0 as Real)..1.0, 0.0..1.0, 0.0..1.0]
            .prop_filter("degenerate all-zero weights", |w| w[0] + w[1] + w[2] > 1e-2)
            .prop_map(|w| {
                let s = w[0] + w[1] + w[2];
                [w[0] / s, w[1] / s, w[2] / s]
            })
    }

    proptest! {
        /// Barycentric coordinates invert the convex combination in 2D.
        #[test]
        fn prop_barycentric_round_trip(
            c0 in vertex(), c1 in vertex(), c2 in vertex(), w in weights(),
        ) {
            // Keep the 2x2 inverse well-conditioned.
            prop_assume!(area2(c0, c1, c2) > 100.0);
            let tri = Triangle::new(c0, c1, c2, 0).unwrap();

            let p = Point2::from(c0.coords * w[0] + c1.coords * w[1] + c2.coords * w[2]);
            let b = tri.barycentric_coordinates(&p);

            prop_assert!((b.x - w[0]).abs() < 1e-2, "l0 {} != {}", b.x, w[0]);
            prop_assert!((b.y - w[1]).abs() < 1e-2, "l1 {} != {}", b.y, w[1]);
            prop_assert!((b.z - w[2]).abs() < 1e-2, "l2 {} != {}", b.z, w[2]);
        }

        /// Any convex combination of the vertices is contained, and the AABB
        /// encloses it.
        #[test]
        fn prop_convex_combinations_are_contained(
            c0 in vertex(), c1 in vertex(), c2 in vertex(), w in weights(),
        ) {
            prop_assume!(area2(c0, c1, c2) > 100.0);
            let tri = Triangle::new(c0, c1, c2, 0).unwrap();

            let p = Point2::from(c0.coords * w[0] + c1.coords * w[1] + c2.coords * w[2]);
            prop_assert!(tri.contains(&p).is_some());

            let aabb = tri.aabb();
            prop_assert!(p.x >= aabb.min.x && p.x <= aabb.max.x);
            prop_assert!(p.y >= aabb.min.y && p.y <= aabb.max.y);
        }

        /// `contains` must agree with the sign of the reported coordinates.
        ///
        /// `contains` re-implements the matrix multiply by hand (indexing
        /// `inv_matrix` directly), so nothing else ties the two together — a
        /// transposed index pair there would go unnoticed.
        #[test]
        fn prop_contains_agrees_with_barycentric_coordinates(
            c0 in vertex(), c1 in vertex(), c2 in vertex(), p in vertex(),
        ) {
            prop_assume!(area2(c0, c1, c2) > 100.0);
            let tri = Triangle::new(c0, c1, c2, 0).unwrap();
            let b = tri.barycentric_coordinates(&p);

            if b.x > CONTAINMENT_EPS && b.y > CONTAINMENT_EPS && b.z > CONTAINMENT_EPS {
                prop_assert!(tri.contains(&p).is_some());
            } else if b.x < -CONTAINMENT_EPS || b.y < -CONTAINMENT_EPS || b.z < -CONTAINMENT_EPS {
                prop_assert!(tri.contains(&p).is_none());
            }
        }
    }

    /// Collinear vertices have no inverse.  This used to panic (`.expect`),
    /// which took the whole process down for one bad face in a real mesh.
    #[test]
    fn test_new_rejects_collinear_vertices() {
        let t = Triangle::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(2.0, 2.0),
            0,
        );
        assert!(t.is_none(), "collinear vertices must not build a triangle");
    }

    #[test]
    fn test_new_rejects_duplicate_vertices() {
        let t = Triangle::new(
            Point2::new(3.0, 4.0),
            Point2::new(3.0, 4.0),
            Point2::new(9.0, 1.0),
            0,
        );
        assert!(t.is_none(), "repeated vertices must not build a triangle");
    }

    /// The vertices map to the unit basis vectors, in the documented order.
    #[test]
    fn test_vertices_map_to_basis_vectors() {
        let c0 = Point2::new(1.0, 1.0);
        let c1 = Point2::new(5.0, 2.0);
        let c2 = Point2::new(2.0, 7.0);
        let tri = Triangle::new(c0, c1, c2, 0).unwrap();

        for (i, v) in [c0, c1, c2].iter().enumerate() {
            let b = tri.barycentric_coordinates(v);
            let got = [b.x, b.y, b.z];
            for (j, g) in got.iter().enumerate() {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (g - want).abs() < 1e-5,
                    "vertex {i} coord {j}: {g} != {want}"
                );
            }
        }
    }

    #[test]
    fn test_contains_rejects_point_outside() {
        let tri = Triangle::new(
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(0.0, 1.0),
            7,
        )
        .unwrap();
        assert!(tri.contains(&Point2::new(0.25, 0.25)).is_some());
        assert!(tri.contains(&Point2::new(0.9, 0.9)).is_none());
        assert!(tri.contains(&Point2::new(-0.5, 0.5)).is_none());
        assert_eq!(tri.contains(&Point2::new(0.25, 0.25)).unwrap().id, 7);
    }
}
