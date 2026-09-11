//! Signed distance from a point to a closed polygonal coastline.
//!
//! The sign is the interesting part.  Distance to the nearest segment is a
//! nearest-neighbour query, but it says nothing about which side of the
//! coastline a point lies on, and the layers downstream need onshore and
//! offshore to differ in sign.  So a query does two traversals of one 2-D BVH
//! over the segments:
//!
//! 1. [`Bvh::nearest_to`] for the unsigned distance;
//! 2. a horizontal ray cast east from the point, counting how many segments it
//!    crosses.  An odd count means the point is inside.
//!
//! Both traversals prune against the same tree, so a coastline of a few
//! hundred thousand segments costs a handful of AABB tests per point rather
//! than a linear scan.

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::{BHShape, BoundingHierarchy},
    bvh::Bvh,
    point_query::PointDistance,
};
use deepsize::{Context, DeepSizeOf};
use nalgebra::Point2;
use rayon::prelude::*;

use crate::real::Real;

/// How far past the eastmost segment a cast ray is extended.
///
/// The ray only has to leave the polygon; any margin does.  Ten metres is
/// comfortably more than the floating-point slack on a projected coordinate.
const RAY_MARGIN: Real = 10.0;

/// One edge of the coastline.
#[derive(Clone, Copy, Debug)]
pub struct Segment {
    a: Point2<Real>,
    b: Point2<Real>,
    node_index: usize,
}

impl Segment {
    pub fn new(a: Point2<Real>, b: Point2<Real>) -> Self {
        Self {
            a,
            b,
            node_index: 0,
        }
    }

    /// Does a ray running east from *point* cross this segment?
    ///
    /// The half-open test on the y interval is a tie-breaker, not an
    /// off-by-one: a ray passing exactly through a shared vertex would
    /// otherwise cross both incident segments and the parity would report an
    /// interior point as exterior.  Counting only the segment whose lower
    /// endpoint is the vertex fixes that.
    #[inline]
    fn crossed_by_eastward_ray(&self, point: Point2<Real>) -> bool {
        let (lo, hi) = if self.a.y < self.b.y {
            (self.a.y, self.b.y)
        } else {
            (self.b.y, self.a.y)
        };
        if point.y < lo || point.y >= hi {
            return false;
        }
        // `hi > lo` follows from the test above, so `b.y - a.y` is non-zero.
        let x = self.a.x + (point.y - self.a.y) * (self.b.x - self.a.x) / (self.b.y - self.a.y);
        x > point.x
    }
}

impl Bounded<Real, 2> for Segment {
    fn aabb(&self) -> Aabb<Real, 2> {
        Aabb::empty().grow(&self.a).grow(&self.b)
    }
}

impl BHShape<Real, 2> for Segment {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

impl PointDistance<Real, 2> for Segment {
    fn distance_squared(&self, point: Point2<Real>) -> Real {
        let edge = self.b - self.a;
        let length_squared = edge.norm_squared();
        // A zero-length segment has no direction to project onto; the distance
        // to either endpoint is the answer.
        let closest = if length_squared > 0.0 {
            let t = ((point - self.a).dot(&edge) / length_squared).clamp(0.0, 1.0);
            self.a + edge * t
        } else {
            self.a
        };
        (point - closest).norm_squared()
    }
}

impl DeepSizeOf for Segment {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

/// A closed polygonal coastline, indexed for signed-distance queries.
pub struct Coastline {
    bvh: Bvh<Real, 2>,
    segments: Vec<Segment>,
    /// X coordinate every cast ray runs out to.
    ray_end_x: Real,
}

impl Coastline {
    /// Index *segments* for querying.
    ///
    /// The segments are expected to form one or more closed rings; the ray
    /// parity test is meaningless otherwise.
    pub fn new(mut segments: Vec<Segment>) -> Self {
        let ray_end_x = segments
            .iter()
            .flat_map(|s| [s.a.x, s.b.x])
            .fold(Real::NEG_INFINITY, Real::max)
            + RAY_MARGIN;
        let bvh = Bvh::build_par(&mut segments);
        Self {
            bvh,
            segments,
            ray_end_x,
        }
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Is *point* inside the coastline?
    pub fn contains(&self, point: Point2<Real>) -> bool {
        if self.segments.is_empty() {
            return false;
        }
        // Only segments whose bounds meet the ray's bounds can be crossed, so
        // the AABB query is a superset of the true crossings and the exact
        // test below decides the rest.
        let ray = Aabb::with_bounds(
            Point2::new(point.x.min(self.ray_end_x), point.y),
            Point2::new(point.x.max(self.ray_end_x), point.y),
        );
        self.bvh
            .traverse_iterator(&ray, &self.segments)
            .filter(|segment| segment.crossed_by_eastward_ray(point))
            .count()
            % 2
            == 1
    }

    /// Distance from *point* to the nearest segment, negative inside.
    ///
    /// Returns [`Real::INFINITY`] for an empty coastline, which keeps every
    /// point unambiguously offshore rather than silently onshore.
    pub fn signed_distance(&self, point: Point2<Real>) -> Real {
        match self.bvh.nearest_to(point, &self.segments) {
            None => Real::INFINITY,
            Some((_, distance)) => {
                if self.contains(point) {
                    -distance
                } else {
                    distance
                }
            }
        }
    }

    /// [`Coastline::signed_distance`] over many points, in parallel.
    pub fn signed_distance_many(&self, x: &[Real], y: &[Real], out: &mut [Real]) {
        out.par_iter_mut()
            .zip(x.par_iter().zip(y.par_iter()))
            .for_each(|(out, (&x, &y))| {
                *out = self.signed_distance(Point2::new(x, y));
            });
    }
}

impl DeepSizeOf for Coastline {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.segments.deep_size_of_children(context)
            + self.bvh.nodes.capacity() * size_of::<bvh::bvh::BvhNode<Real, 2>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Close a ring of points into segments.
    fn ring(points: &[(Real, Real)]) -> Vec<Segment> {
        points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|(&(ax, ay), &(bx, by))| Segment::new(Point2::new(ax, ay), Point2::new(bx, by)))
            .collect()
    }

    /// The unit square, which gives every test below an analytic oracle:
    /// the signed distance to `[0, 1]^2` is known in closed form.
    fn unit_square() -> Coastline {
        Coastline::new(ring(&[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]))
    }

    /// Signed distance to the unit square, worked out directly.
    fn square_oracle(x: Real, y: Real) -> Real {
        let dx = (0.0 - x).max(x - 1.0);
        let dy = (0.0 - y).max(y - 1.0);
        if dx <= 0.0 && dy <= 0.0 {
            // Inside: distance to the closest edge, negated.
            -(-dx).min(-dy)
        } else {
            dx.max(0.0).hypot(dy.max(0.0))
        }
    }

    #[test]
    fn inside_points_are_contained() {
        let c = unit_square();
        for (x, y) in [(0.5, 0.5), (0.1, 0.9), (0.99, 0.01)] {
            assert!(c.contains(Point2::new(x, y)), "({x}, {y}) should be inside");
        }
    }

    #[test]
    fn outside_points_are_not_contained() {
        let c = unit_square();
        for (x, y) in [(-0.5, 0.5), (1.5, 0.5), (0.5, -0.5), (0.5, 1.5), (2.0, 2.0)] {
            assert!(
                !c.contains(Point2::new(x, y)),
                "({x}, {y}) should be outside"
            );
        }
    }

    /// A ray through a shared vertex must not be counted twice.  Every corner
    /// of this diamond sits at a y where two edges meet, so a ray cast at that
    /// exact height crosses a vertex.
    #[test]
    fn a_ray_through_a_vertex_counts_once() {
        let c = Coastline::new(ring(&[(0.0, 0.0), (1.0, -1.0), (2.0, 0.0), (1.0, 1.0)]));
        assert!(c.contains(Point2::new(1.0, 0.0)), "centre is inside");
        assert!(!c.contains(Point2::new(-1.0, 0.0)), "due west is outside");
        assert!(!c.contains(Point2::new(3.0, 0.0)), "due east is outside");
    }

    #[test]
    fn signed_distance_matches_the_oracle() {
        let c = unit_square();
        for (x, y) in [
            (0.5, 0.5),
            (0.25, 0.5),
            (-1.0, 0.5),
            (2.0, 0.5),
            (0.5, 2.0),
            (-1.0, -1.0),
        ] {
            let got = c.signed_distance(Point2::new(x, y));
            let want = square_oracle(x, y);
            assert!((got - want).abs() < 1e-4, "at ({x}, {y}): {got} != {want}");
        }
    }

    #[test]
    fn an_empty_coastline_puts_everything_offshore() {
        let c = Coastline::new(Vec::new());
        assert!(c.is_empty());
        assert!(!c.contains(Point2::new(0.0, 0.0)));
        assert_eq!(c.signed_distance(Point2::new(0.0, 0.0)), Real::INFINITY);
    }

    #[test]
    fn a_degenerate_segment_has_a_finite_distance() {
        // A zero-length segment has no direction to project onto.
        let c = Coastline::new(vec![Segment::new(
            Point2::new(1.0, 1.0),
            Point2::new(1.0, 1.0),
        )]);
        let d = c.signed_distance(Point2::new(4.0, 5.0));
        assert!((d - 5.0).abs() < 1e-4, "{d}");
    }

    #[test]
    fn many_matches_one_at_a_time() {
        let c = unit_square();
        let x = [0.5, -1.0, 2.0, 0.25];
        let y = [0.5, 0.5, 0.5, 0.75];
        let mut out = [0.0; 4];
        c.signed_distance_many(&x, &y, &mut out);
        for i in 0..4 {
            let want = c.signed_distance(Point2::new(x[i], y[i]));
            assert_eq!(out[i], want);
        }
    }

    proptest! {
        /// The sign follows containment, and the magnitude follows the oracle,
        /// everywhere in and around the square.
        #[test]
        fn prop_signed_distance_matches_the_oracle(
            x in (-2.0 as Real)..3.0,
            y in (-2.0 as Real)..3.0,
        ) {
            let c = unit_square();
            let got = c.signed_distance(Point2::new(x, y));
            let want = square_oracle(x, y);
            prop_assert!((got - want).abs() < 1e-4, "at ({x}, {y}): {got} != {want}");
        }

        /// Containment agrees with the sign of the distance.
        #[test]
        fn prop_sign_agrees_with_containment(
            x in (-2.0 as Real)..3.0,
            y in (-2.0 as Real)..3.0,
        ) {
            let c = unit_square();
            let inside = c.contains(Point2::new(x, y));
            let signed = c.signed_distance(Point2::new(x, y));
            prop_assert_eq!(inside, signed < 0.0);
        }
    }
}
