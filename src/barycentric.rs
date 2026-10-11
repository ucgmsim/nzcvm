//! Barycentric coordinates and interpolation.
//!
//! Every piecewise-linear interpolant in the crate (qualities and depths
//! over tetrahedra, elevations over surface triangles, the TIN's fit of a
//! grid) is a weighted sum of corner values, [`interpolate`], with weights
//! from [`triangle_coordinates`] or a simplex's own precomputed inverse.

use nalgebra::{Point2, RealField};

/// The sum of `values` weighted by barycentric `weights`.
#[inline(always)]
pub fn interpolate<T: RealField + Copy, const N: usize>(values: [T; N], weights: [T; N]) -> T {
    values
        .into_iter()
        .zip(weights)
        .fold(T::zero(), |sum, (value, weight)| sum + weight * value)
}

/// The barycentric coordinates of `p` in the triangle `corners`, in corner
/// order.  Non-finite if the triangle is degenerate.
pub fn triangle_coordinates<T: RealField + Copy>(corners: [Point2<T>; 3], p: Point2<T>) -> [T; 3] {
    let [a, b, c] = corners;
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let det = ab.perp(&ac);
    let lb = ap.perp(&ac) / det;
    let lc = ab.perp(&ap) / det;
    [T::one() - lb - lc, lb, lc]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn point() -> impl Strategy<Value = Point2<f64>> {
        (-50.0..50.0, -50.0..50.0).prop_map(|(x, y)| Point2::new(x, y))
    }

    proptest! {
        #[test]
        fn prop_coordinates_reproduce_the_point_and_sum_to_one(
            a in point(), b in point(), c in point(), p in point(),
        ) {
            prop_assume!(((b - a).perp(&(c - a))).abs() > 1.0);
            let w = triangle_coordinates([a, b, c], p);
            prop_assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            let x = interpolate([a.x, b.x, c.x], w);
            let y = interpolate([a.y, b.y, c.y], w);
            prop_assert!((x - p.x).abs() < 1e-6 && (y - p.y).abs() < 1e-6);
        }

        #[test]
        fn prop_interpolation_reproduces_a_plane(
            a in point(), b in point(), c in point(), p in point(),
            slope in (-5.0..5.0, -5.0..5.0), offset in -100.0..100.0,
        ) {
            prop_assume!(((b - a).perp(&(c - a))).abs() > 1.0);
            let plane = |q: Point2<f64>| slope.0 * q.x + slope.1 * q.y + offset;
            let value = interpolate([a, b, c].map(plane), triangle_coordinates([a, b, c], p));
            prop_assert!((value - plane(p)).abs() < 1e-6 * (1.0 + plane(p).abs()));
        }
    }

    #[test]
    fn test_corners_have_unit_coordinates() {
        let corners = [
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(0.0, 3.0),
        ];
        for (i, &corner) in corners.iter().enumerate() {
            let w = triangle_coordinates(corners, corner);
            for (j, &wj) in w.iter().enumerate() {
                assert_eq!(wj, if i == j { 1.0 } else { 0.0 });
            }
        }
    }
}
