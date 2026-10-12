//! Interpolation on structured grids in a Cartesian frame.
//!
//! In a structured grid, node `[j, k]` neighbours nodes `[j ± 1, k]` and
//! `[j, k ± 1]`, but the cells can be arbitrary quadrilaterals. A regular
//! latitude/longitude grid projected into a Cartesian frame is one example.
//! [`StructuredGrid`] interpolates bilinearly within those quadrilaterals.
//! Outside the grid, it clamps the query point onto the edge cell, which
//! extends the edge row or column outward.
//!
//! [`StructuredGrid::query_triangulated`] instead splits each cell into two
//! triangles along its `[j, k]`–`[j + 1, k + 1]` diagonal and interpolates
//! barycentrically, as a triangle mesh over the nodes would. It finds the
//! cell the same way as the bilinear query, so it needs no search tree.

use nalgebra::{Matrix2, Vector2};
use ndarray::Array2;

/// Tolerance on the local cell coordinates when deciding a point is outside.
const OUTSIDE_TOLERANCE: f64 = 1e-9;

/// Most negative barycentric weight a point can have and still count as
/// inside a triangle in [`StructuredGrid::query_triangulated`].
const TRIANGLE_TOLERANCE: f64 = 1e-4;

/// A structured grid of Cartesian nodes carrying one value per node.
pub struct StructuredGrid {
    x: Array2<f64>,
    y: Array2<f64>,
    values: Array2<f64>,
    /// Origin and inverse basis of an affine map from coordinates to
    /// fractional `(j, k)` index, used to guess the cell containing a point.
    origin: Vector2<f64>,
    inverse_basis: Matrix2<f64>,
}

impl StructuredGrid {
    /// Create a grid from `(m, n)` arrays of node coordinates and values.
    ///
    /// Returns an error if the shapes differ, the grid has fewer than two
    /// rows or columns, or its corner nodes don't span a 2D area.
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
        let node = |j: usize, k: usize| Vector2::new(x[[j, k]], y[[j, k]]);
        let origin = node(0, 0);
        let row_step = (node(m - 1, 0) - origin) / (m - 1) as f64;
        let col_step = (node(0, n - 1) - origin) / (n - 1) as f64;
        let inverse_basis = Matrix2::from_columns(&[row_step, col_step])
            .try_inverse()
            .ok_or("grid corner nodes are collinear")?;
        Ok(Self {
            x,
            y,
            values,
            origin,
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
        // The cells are nearly parallelograms, so the quadratic coefficient is
        // nearly zero. This form of the smaller root avoids the cancellation
        // the textbook formula suffers there, and reduces to the linear
        // solution when the cell is a parallelogram.
        let discriminant = (linear * linear - 4.0 * quadratic * constant).max(0.0);
        let root = -0.5 * (linear + linear.signum() * discriminant.sqrt());
        let t = constant / root;
        let d = b + t * c;
        let s = (q - t * a).dot(&d) / d.norm_squared();
        (t, s)
    }

    /// The cell `(j, k)` containing `p`, or the edge cell nearest it when
    /// `p` lies outside the grid, and the local coordinates `(t, s)` of `p`
    /// in that cell.
    fn locate(&self, p: Vector2<f64>) -> (usize, usize, f64, f64) {
        let (m, n) = self.x.dim();
        let guess = self.inverse_basis * (p - self.origin);
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
        (j, k, t, s)
    }

    /// Interpolate the grid bilinearly at the point `(px, py)`.
    ///
    /// Returns the interpolated value and whether the point lies outside
    /// the grid. An outside point is clamped onto the edge cell when `clamp`
    /// is true, and takes NaN otherwise.
    pub fn query(&self, px: f64, py: f64, clamp: bool) -> (f64, bool) {
        let (j, k, t, s) = self.locate(Vector2::new(px, py));
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

    /// Interpolate the grid barycentrically at the point `(px, py)`, over
    /// the two triangles of each cell either side of its
    /// `[j, k]`–`[j + 1, k + 1]` diagonal.
    ///
    /// A cell's edges are straight in both the bilinear and the triangulated
    /// view, so the cell [`Self::locate`] finds is the one whose triangles
    /// contain the point. Of the two, the point takes the triangle it is
    /// furthest inside, which is the one containing it. It returns NaN when
    /// neither triangle contains it to within [`TRIANGLE_TOLERANCE`].
    pub fn query_triangulated(&self, px: f64, py: f64) -> f64 {
        let p = Vector2::new(px, py);
        let (j, k, _, _) = self.locate(p);
        let corner = self.node(j, k);
        let diagonal = self.node(j + 1, k + 1);
        let mut best = (f64::NEG_INFINITY, f64::NAN);
        for (side_j, side_k) in [(j + 1, k), (j, k + 1)] {
            let side = self.node(side_j, side_k);
            let Some([w_corner, w_side, w_diagonal]) =
                barycentric_weights(corner, side, diagonal, p)
            else {
                continue;
            };
            let worst = w_corner.min(w_side).min(w_diagonal);
            if worst > best.0 {
                let value = w_corner * self.values[[j, k]]
                    + w_side * self.values[[side_j, side_k]]
                    + w_diagonal * self.values[[j + 1, k + 1]];
                best = (worst, value);
            }
        }
        if best.0 >= -TRIANGLE_TOLERANCE {
            best.1
        } else {
            f64::NAN
        }
    }
}

/// Barycentric weights of `p` in the triangle `(a, b, c)`, or `None` if the
/// triangle has no area.
fn barycentric_weights(
    a: Vector2<f64>,
    b: Vector2<f64>,
    c: Vector2<f64>,
    p: Vector2<f64>,
) -> Option<[f64; 3]> {
    let basis = Matrix2::from_columns(&[b - a, c - a]).try_inverse()?;
    let w = basis * (p - a);
    Some([1.0 - w.x - w.y, w.x, w.y])
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

    /// A field linear in x and y, which barycentric interpolation reproduces
    /// exactly on any triangulation of the nodes.
    fn plane(x: f64, y: f64) -> f64 {
        3e-3 * x - 2e-3 * y + 5.0
    }

    fn plane_grid(m: usize, n: usize) -> StructuredGrid {
        let x = Array2::from_shape_fn((m, n), |(j, k)| position(j as f64, k as f64).0);
        let y = Array2::from_shape_fn((m, n), |(j, k)| position(j as f64, k as f64).1);
        let values = Array2::from_shape_fn((m, n), |(j, k)| plane(x[[j, k]], y[[j, k]]));
        StructuredGrid::new(x, y, values).unwrap()
    }

    /// A unit cell with a value only at node `[1, 0]`. The `[0, 0]`–`[1, 1]`
    /// diagonal puts that node in one triangle only, and a bilinear or
    /// other-diagonal interpolation would give different values.
    #[test]
    fn test_triangulated_splits_along_the_main_diagonal() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let values = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let g = StructuredGrid::new(x, y, values).unwrap();
        // (0.75, 0.25) lies in triangle [0, 0], [1, 0], [1, 1], with weight
        // 0.5 on node [1, 0]. Bilinear interpolation would give 0.5625.
        assert!((g.query_triangulated(0.75, 0.25) - 0.5).abs() < 1e-12);
        // (0.25, 0.75) lies in triangle [0, 0], [0, 1], [1, 1].
        assert!(g.query_triangulated(0.25, 0.75).abs() < 1e-12);
        // On the diagonal both triangles agree.
        assert!(g.query_triangulated(0.5, 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_triangulated_reproduces_nodes() {
        let g = plane_grid(6, 5);
        for j in 0..6 {
            for k in 0..5 {
                let (px, py) = position(j as f64, k as f64);
                assert!((g.query_triangulated(px, py) - plane(px, py)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_triangulated_outside_is_nan() {
        let g = plane_grid(4, 4);
        for (j, k) in [
            (-0.5, 1.5),
            (1.5, -0.5),
            (3.5, 1.0),
            (1.0, 3.5),
            (-1.0, -1.0),
        ] {
            let (px, py) = position(j, k);
            assert!(g.query_triangulated(px, py).is_nan(), "({j}, {k})");
        }
    }

    #[test]
    fn test_triangulated_propagates_nan_values() {
        let x = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = Array2::from_shape_vec((2, 2), vec![0.0, 1.0, 0.0, 1.0]).unwrap();
        let values = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, f64::NAN, 0.0]).unwrap();
        let g = StructuredGrid::new(x, y, values).unwrap();
        assert!(g.query_triangulated(0.75, 0.25).is_nan());
        assert_eq!(g.query_triangulated(0.25, 0.75), 0.0);
    }

    #[test]
    fn test_rejects_mismatched_shapes() {
        let a = Array2::<f64>::zeros((3, 3));
        let b = Array2::<f64>::zeros((3, 4));
        assert!(StructuredGrid::new(a.clone(), a, b).is_err());
    }

    proptest! {
        /// Barycentric interpolation reproduces a field linear in x and y
        /// wherever the point falls in the grid.
        #[test]
        fn prop_triangulated_reproduces_linear_field(j in 0.0f64..7.0, k in 0.0f64..5.0) {
            let g = plane_grid(8, 6);
            let (px, py) = position(j, k);
            let got = g.query_triangulated(px, py);
            prop_assert!((got - plane(px, py)).abs() < 1e-9, "{got} != {}", plane(px, py));
        }

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
    }
}
