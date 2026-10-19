//! Bilinear interpolation on structured grids in a Cartesian frame.
//!
//! In a structured grid, node `[j, k]` neighbours nodes `[j ± 1, k]` and
//! `[j, k ± 1]`, but the cells can be arbitrary quadrilaterals. A regular
//! latitude/longitude grid projected into a Cartesian frame is one example.
//! [`StructuredGrid`] interpolates bilinearly within those quadrilaterals.
//! Outside the grid, it clamps the query point onto the edge cell, which
//! extends the edge row or column outward.

use nalgebra::{Matrix2, Vector2};
use ndarray::Array2;

/// Tolerance on the local cell coordinates when deciding a point is outside.
const OUTSIDE_TOLERANCE: f64 = 1e-9;

/// A structured grid of Cartesian nodes carrying one value per node.
pub struct StructuredGrid {
    x: Array2<f64>,
    y: Array2<f64>,
    values: Array2<f64>,
    /// Affine map from coordinates to fractional `(j, k)` index, used to
    /// guess the cell containing a point: `index_centre + inverse_basis *
    /// (p - centre)`.
    centre: Vector2<f64>,
    index_centre: Vector2<f64>,
    inverse_basis: Matrix2<f64>,
}

impl StructuredGrid {
    /// Create a grid from `(m, n)` arrays of node coordinates and values.
    ///
    /// Returns an error if the shapes differ, the grid has fewer than two
    /// rows or columns, or its nodes don't span a 2D area.
    pub fn new(x: Array2<f64>, y: Array2<f64>, values: Array2<f64>) -> Result<Self, String> {
        let (m, n) = x.dim();
        if y.dim() != (m, n) || values.dim() != (m, n) {
            return Err(format!(
                "x, y and values must have the same shape; got {:?}, {:?} and {:?}",
                x.dim(),
                y.dim(),
                values.dim(),
            ));
        }
        if m < 2 || n < 2 {
            return Err(format!(
                "grid must have at least two rows and columns; got {m}x{n}"
            ));
        }
        // Fit the index guess to every node by least squares, so curved or
        // unevenly spaced rows and columns pull it no further than they must.
        let count = (m * n) as f64;
        let centre = Vector2::new(x.sum() / count, y.sum() / count);
        let index_centre = Vector2::new((m - 1) as f64 / 2.0, (n - 1) as f64 / 2.0);
        let mut position_moments = Matrix2::zeros();
        let mut index_moments = Matrix2::zeros();
        for ((j, k), &node_x) in x.indexed_iter() {
            let position = Vector2::new(node_x, y[[j, k]]) - centre;
            let index = Vector2::new(j as f64, k as f64) - index_centre;
            position_moments += position * position.transpose();
            index_moments += index * position.transpose();
        }
        let inverse_basis = index_moments
            * position_moments
                .try_inverse()
                .ok_or("grid nodes are collinear")?;
        Ok(Self {
            x,
            y,
            values,
            centre,
            index_centre,
            inverse_basis,
        })
    }

    fn node(&self, j: usize, k: usize) -> Vector2<f64> {
        Vector2::new(self.x[[j, k]], self.y[[j, k]])
    }

    /// Local coordinates `(t, s)` of `p` in cell `(j, k)`, where `t` runs
    /// from row `j` to `j + 1` and `s` from column `k` to `k + 1`.
    ///
    /// Inverts the cell's bilinear map in closed form. Points outside the
    /// cell give coordinates outside `[0, 1]`.
    fn local_coordinates(&self, j: usize, k: usize, p: Vector2<f64>) -> (f64, f64) {
        let p00 = self.node(j, k);
        let a = self.node(j + 1, k) - p00;
        let b = self.node(j, k + 1) - p00;
        let c = self.node(j + 1, k + 1) - p00 - a - b;
        let q = p - p00;
        // q = t a + s b + t s c. Crossing both sides with b + t c eliminates
        // s, leaving (a × c) t² + (a × b - q × c) t - q × b = 0.
        let cross = |u: Vector2<f64>, v: Vector2<f64>| u.x * v.y - u.y * v.x;
        let quadratic = cross(a, c);
        let linear = cross(a, b) - cross(q, c);
        let constant = -cross(q, b);
        // Both roots in the stable pair: `constant / root` avoids the
        // cancellation the textbook formula suffers when the cell is nearly a
        // parallelogram, and reduces to the linear solution when it is one.
        // The other root, `root / quadratic`, exists only for a curved cell.
        let discriminant = (linear * linear - 4.0 * quadratic * constant).max(0.0);
        let root = -0.5 * (linear + linear.signum() * discriminant.sqrt());
        let solve = |t: f64| {
            let d = b + t * c;
            let s = (q - t * a).dot(&d) / d.norm_squared();
            (t, s)
        };
        // The root inside the cell is the one we want. Points outside it may
        // lie outside both, so keep whichever pair is closest to the cell. A
        // root where the cell's cross-section collapses (`b + t c = 0`)
        // leaves `s` undefined and is never chosen over a finite one.
        let excess = |v: f64| (-v).max(v - 1.0).max(0.0);
        let distance = |(t, s): (f64, f64)| {
            let distance = excess(t) + excess(s);
            if distance.is_nan() {
                f64::INFINITY
            } else {
                distance
            }
        };
        let first = solve(constant / root);
        if quadratic == 0.0 {
            return first;
        }
        let second = solve(root / quadratic);
        if distance(second) < distance(first) {
            second
        } else {
            first
        }
    }

    /// Interpolate the grid bilinearly at the point `(px, py)`.
    ///
    /// Returns the interpolated value and whether the point lies outside
    /// the grid. An outside point is clamped onto the edge cell when `clamp`
    /// is true, and takes NaN otherwise.
    pub fn query(&self, px: f64, py: f64, clamp: bool) -> (f64, bool) {
        let (m, n) = self.x.dim();
        let p = Vector2::new(px, py);
        let guess = self.index_centre + self.inverse_basis * (p - self.centre);
        let to_cell = |v: f64, len: usize| (v.floor().max(0.0) as usize).min(len - 2);
        let mut j = to_cell(guess.x, m);
        let mut k = to_cell(guess.y, n);

        // Walk from the guessed cell towards the cell containing p. Each step
        // moves one cell, and the affine guess is close, so few steps run.
        let (mut t, mut s) = self.local_coordinates(j, k, p);
        for _ in 0..(m + n) {
            let (next_j, next_k) = (step_index(j, t, m - 2), step_index(k, s, n - 2));
            if (next_j, next_k) == (j, k) {
                break;
            }
            (j, k) = (next_j, next_k);
            (t, s) = self.local_coordinates(j, k, p);
        }

        let inside = |v: f64| (-OUTSIDE_TOLERANCE..=1.0 + OUTSIDE_TOLERANCE).contains(&v);
        let outside = !inside(t) || !inside(s);
        if outside && !clamp {
            return (f64::NAN, true);
        }
        let t = t.clamp(0.0, 1.0);
        let s = s.clamp(0.0, 1.0);
        let value = (1.0 - t) * (1.0 - s) * self.values[[j, k]]
            + (1.0 - t) * s * self.values[[j, k + 1]]
            + t * (1.0 - s) * self.values[[j + 1, k]]
            + t * s * self.values[[j + 1, k + 1]];
        (value, outside)
    }
}

/// The neighbouring cell index in the direction of local coordinate `v`,
/// staying within `[0, max]`.
fn step_index(index: usize, v: f64, max: usize) -> usize {
    if v < -OUTSIDE_TOLERANCE && index > 0 {
        index - 1
    } else if v > 1.0 + OUTSIDE_TOLERANCE && index < max {
        index + 1
    } else {
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A bilinear field in grid index space, which bilinear interpolation
    /// reproduces exactly at any point of a mapped grid.
    fn field(j: f64, k: f64) -> f64 {
        3.0 * j - 2.0 * k + 0.5 * j * k + 5.0
    }

    /// Bilinear map from index space to a rotated and sheared plane. The `j * k` terms make every cell a different non-rectangular
    /// quadrilateral, while local cell coordinates stay equal to the
    /// fractional index, which gives the tests an exact oracle.
    fn position(j: f64, k: f64) -> (f64, f64) {
        (
            1.5e6 + 400.0 * k + 60.0 * j + 2.0 * j * k,
            5.2e6 + 450.0 * j - 50.0 * k + 1.5 * j * k,
        )
    }

    fn grid(m: usize, n: usize) -> StructuredGrid {
        let x = Array2::from_shape_fn((m, n), |(j, k)| position(j as f64, k as f64).0);
        let y = Array2::from_shape_fn((m, n), |(j, k)| position(j as f64, k as f64).1);
        let values = Array2::from_shape_fn((m, n), |(j, k)| field(j as f64, k as f64));
        StructuredGrid::new(x, y, values).unwrap()
    }

    #[test]
    fn test_reproduces_grid_nodes() {
        let g = grid(6, 5);
        for j in 0..6 {
            for k in 0..5 {
                let (px, py) = position(j as f64, k as f64);
                let (got, outside) = g.query(px, py, false);
                assert!(!outside, "node ({j}, {k}) reported outside");
                assert!((got - field(j as f64, k as f64)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_outside_without_clamp_is_nan() {
        let g = grid(4, 4);
        let (px, py) = position(-3.0, 1.5);
        let (got, outside) = g.query(px, py, false);
        assert!(outside);
        assert!(got.is_nan());
    }

    #[test]
    fn test_clamps_outside_onto_edge() {
        // On an affine grid the edge extension is exact: a point beyond the
        // last row takes the value on that row at the same column position.
        let x = Array2::from_shape_fn((3, 3), |(_, k)| 100.0 * k as f64);
        let y = Array2::from_shape_fn((3, 3), |(j, _)| 100.0 * j as f64);
        let values = Array2::from_shape_fn((3, 3), |(j, k)| field(j as f64, k as f64));
        let g = StructuredGrid::new(x, y, values).unwrap();
        let (got, outside) = g.query(150.0, 900.0, true);
        assert!(outside);
        assert!((got - field(2.0, 1.5)).abs() < 1e-9);
        let (got, outside) = g.query(-50.0, -50.0, true);
        assert!(outside);
        assert!((got - field(0.0, 0.0)).abs() < 1e-9);
    }

    /// A cell far from a parallelogram, to check the closed-form inversion
    /// picks the right root of its quadratic.
    #[test]
    fn test_inverts_trapezoid_cell() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 100.0, 30.0, 70.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 80.0, 90.0]).unwrap();
        let values = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 7.0]).unwrap();
        let g = StructuredGrid::new(x.clone(), y.clone(), values.clone()).unwrap();
        for (t, s) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.95), (0.3, 0.8), (1.0, 0.0)] {
            let weights = [(1.0 - t) * (1.0 - s), (1.0 - t) * s, t * (1.0 - s), t * s];
            let blend = |a: &Array2<f64>| {
                weights[0] * a[[0, 0]]
                    + weights[1] * a[[0, 1]]
                    + weights[2] * a[[1, 0]]
                    + weights[3] * a[[1, 1]]
            };
            let (got, outside) = g.query(blend(&x), blend(&y), false);
            assert!(!outside, "({t}, {s}) reported outside");
            assert!((got - blend(&values)).abs() < 1e-9, "at ({t}, {s})");
        }
    }

    /// A trapezoid whose far edge is three times its near edge. The spurious
    /// root is negative and smaller in magnitude than the true one, and at
    /// `s = 0.5` it collapses the cell's cross-section.
    #[test]
    fn test_inverts_widening_trapezoid_cell() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 10.0, 0.0, 30.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        let values = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let g = StructuredGrid::new(x, y, values).unwrap();
        for (t, s) in [(0.2, 0.5), (0.5, 0.5), (0.9, 0.5), (0.9, 0.1), (0.99, 0.99)] {
            let (px, py) = (s * 10.0 + t * s * 20.0, t * 10.0);
            for clamp in [false, true] {
                let (got, outside) = g.query(px, py, clamp);
                assert!(!outside, "({t}, {s}) reported outside");
                assert!((got - t).abs() < 1e-9, "at ({t}, {s}): {got}");
            }
        }
    }

    #[test]
    fn test_rejects_mismatched_shapes() {
        let a = Array2::<f64>::zeros((3, 3));
        let b = Array2::<f64>::zeros((3, 4));
        assert!(StructuredGrid::new(a.clone(), a, b).is_err());
    }

    proptest! {
        /// Bilinear interpolation in each mapped cell reproduces a field that is
        /// bilinear in index space, wherever the point falls in the grid.
        #[test]
        fn prop_reproduces_bilinear_field(j in 0.0f64..7.0, k in 0.0f64..5.0) {
            let g = grid(8, 6);
            let (px, py) = position(j, k);
            let (got, outside) = g.query(px, py, false);
            prop_assert!(!outside);
            prop_assert!((got - field(j, k)).abs() < 1e-6, "{got} != {}", field(j, k));
        }

        /// Every point of a random convex quadrilateral maps back to the local
        /// coordinates it was built from.
        #[test]
        fn prop_inverts_convex_quad(
            far in prop::array::uniform2(-0.9f64..0.9),
            spread in 0.1f64..3.0,
            t in 0.0f64..1.0,
            s in 0.0f64..1.0,
        ) {
            // A unit-height trapezoid-like quad: near edge (0, 0)-(1, 0), far
            // edge from (far[0], 1) to (far[0] + spread, 1) skewed by far[1].
            let corners = [
                (0.0, 0.0),
                (1.0, 0.0),
                (far[0], 1.0),
                (far[0] + spread, 1.0 + far[1]),
            ];
            // Walking the boundary (0, 0) → (0, 1) → (1, 1) → (1, 0), every
            // turn must have the same sign for the quad to be convex.
            let ring = [corners[0], corners[1], corners[3], corners[2]];
            let turns: Vec<f64> = (0..4)
                .map(|i| {
                    let (p, q, r) = (ring[i], ring[(i + 1) % 4], ring[(i + 2) % 4]);
                    (q.0 - p.0) * (r.1 - q.1) - (q.1 - p.1) * (r.0 - q.0)
                })
                .collect();
            prop_assume!(turns.iter().all(|&turn| turn > 1e-3));
            let x = Array2::from_shape_vec((2, 2), corners.iter().map(|c| c.0).collect()).unwrap();
            let y = Array2::from_shape_vec((2, 2), corners.iter().map(|c| c.1).collect()).unwrap();
            let values = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
            let blend = |a: &Array2<f64>| {
                (1.0 - t) * (1.0 - s) * a[[0, 0]]
                    + (1.0 - t) * s * a[[0, 1]]
                    + t * (1.0 - s) * a[[1, 0]]
                    + t * s * a[[1, 1]]
            };
            let (px, py) = (blend(&x), blend(&y));
            let g = StructuredGrid::new(x, y, values).unwrap();
            let (got, outside) = g.query(px, py, false);
            prop_assert!(!outside, "({t}, {s}) reported outside");
            prop_assert!((got - t).abs() < 1e-7, "{got} != {t}");
        }
    }
}
