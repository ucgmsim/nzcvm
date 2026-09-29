//! Bilinear interpolation on structured grids of NZTM nodes.
//!
//! Basin surfaces and mesh sizing fields come from regular latitude/longitude
//! grids. Projected into NZTM, such a grid stays structured: node `[j, k]`
//! neighbours nodes `[j ± 1, k]` and `[j, k ± 1]`, but its cells are
//! arbitrary quadrilaterals. [`StructuredGrid`] interpolates bilinearly
//! within those quadrilaterals, entirely in NZTM coordinates. Outside the
//! grid the query point is clamped onto the edge cell, which extends the edge
//! row or column outward.

use nalgebra::{Matrix2, Vector2};
use ndarray::Array2;

/// Tolerance on the local cell coordinates when deciding a point is outside.
const OUTSIDE_TOLERANCE: f64 = 1e-9;

/// Newton iterations used to invert a cell's bilinear map.
const NEWTON_ITERATIONS: usize = 8;

/// A structured grid of NZTM nodes carrying one value per node.
pub struct StructuredGrid {
    x: Array2<f64>,
    y: Array2<f64>,
    values: Array2<f64>,
    /// Origin and inverse basis of an affine map from NZTM to fractional
    /// `(j, k)` index, used to guess the cell that contains a point.
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
    /// Inverts the cell's bilinear map with Newton's method. Points outside
    /// the cell give coordinates outside `[0, 1]`.
    fn local_coordinates(&self, j: usize, k: usize, p: Vector2<f64>) -> (f64, f64) {
        let p00 = self.node(j, k);
        let p01 = self.node(j, k + 1);
        let p10 = self.node(j + 1, k);
        let p11 = self.node(j + 1, k + 1);
        let (mut t, mut s) = (0.5, 0.5);
        for _ in 0..NEWTON_ITERATIONS {
            let residual = (1.0 - t) * (1.0 - s) * p00
                + (1.0 - t) * s * p01
                + t * (1.0 - s) * p10
                + t * s * p11
                - p;
            let d_dt = (1.0 - s) * (p10 - p00) + s * (p11 - p01);
            let d_ds = (1.0 - t) * (p01 - p00) + t * (p11 - p10);
            let Some(step) = Matrix2::from_columns(&[d_dt, d_ds])
                .try_inverse()
                .map(|inv| inv * residual)
            else {
                break;
            };
            t -= step.x;
            s -= step.y;
            if step.norm() < 1e-12 {
                break;
            }
        }
        (t, s)
    }

    /// Interpolate the grid bilinearly at the NZTM point `(px, py)`.
    ///
    /// Returns the interpolated value and whether the point lies outside
    /// the grid. An outside point is clamped onto the edge cell when `clamp`
    /// is true, and takes NaN otherwise.
    pub fn query(&self, px: f64, py: f64, clamp: bool) -> (f64, bool) {
        let (m, n) = self.x.dim();
        let p = Vector2::new(px, py);
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

    /// Bilinear map from index space to a rotated and sheared NZTM-like
    /// plane. The `j * k` terms make every cell a different non-rectangular
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
    }
}
