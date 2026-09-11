use crate::real::Real;
use deepsize::DeepSizeOf;
use ndarray::{Array1, Array2, ArrayView1};
use serde::Serialize;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Seismic material properties at a single point.
///
/// `alpha` is the opacity weight used when blending overlapping models;
/// it follows the Porter-Duff "over" compositing rule in [`Quality::blend`].
#[derive(
    Clone,
    Debug,
    Copy,
    PartialEq,
    DeepSizeOf,
    Serialize,
    FromBytes,
    IntoBytes,
    KnownLayout,
    Immutable,
)]
#[repr(C)]
pub struct Quality {
    pub rho: Real,
    pub vp: Real,
    pub vs: Real,
    pub qp: Real,
    pub qs: Real,
    /// Opacity weight in `[0, 1]`.  A value of `1.0` means fully opaque:
    /// higher-priority models beneath are ignored.
    pub alpha: Real,
}

/// Opacity at or below which a quality is treated as fully transparent.
///
/// Deliberately pinned to `f32::EPSILON` rather than `Real::EPSILON`: this
/// threshold expresses when an opacity is *physically* negligible, not the
/// precision of the storage type, so it must not shift by nine orders of
/// magnitude when the `high_precision` feature switches `Real` to `f64`.
const TRANSPARENT_ALPHA: Real = f32::EPSILON as Real;

impl Quality {
    /// Composite `self` over `rhs` using the Porter-Duff "over" operator.
    ///
    /// The resulting `alpha` is `self.alpha + rhs.alpha * (1 - self.alpha)`.
    /// Material properties are blended proportionally.
    ///
    /// # Examples
    ///
    /// A fully-opaque quality blended with anything stays unchanged:
    ///
    /// ```
    /// use nzcvm::quality::Quality;
    /// let a = Quality { rho: 2700.0, vp: 6000.0, vs: 3500.0, qp: 200.0, qs: 100.0, alpha: 1.0 };
    /// let b = Quality { rho: 1000.0, vp: 1500.0, vs: 0.0, qp: 50.0, qs: 25.0, alpha: 0.5 };
    /// let blended = a.blend(&b);
    /// assert!((blended.rho - 2700.0).abs() < 1e-3);
    /// assert!((blended.alpha - 1.0).abs() < 1e-3);
    /// ```
    pub fn blend(&self, rhs: &Quality) -> Quality {
        // These shortcuts are required to ensure correct behaviour when
        // interpolating Qp/Qs.
        if self.alpha < TRANSPARENT_ALPHA {
            return *rhs;
        } else if rhs.alpha < TRANSPARENT_ALPHA {
            return *self;
        }
        let alpha = self.alpha + rhs.alpha * (1.0 - self.alpha);
        let a0 = self.alpha / alpha;
        let a1 = rhs.alpha * (1.0 - self.alpha) / alpha;
        let qp = (a0 * self.qp.recip() + a1 * rhs.qp.recip()).recip();
        let qs = (a0 * self.qs.recip() + a1 * rhs.qs.recip()).recip();

        Self {
            rho: a0 * self.rho + a1 * rhs.rho,
            vp: a0 * self.vp + a1 * rhs.vp,
            vs: a0 * self.vs + a1 * rhs.vs,
            qp,
            qs,
            alpha,
        }
    }
}
/// Interpolates a tetrahedral cell's vertex properties.
pub fn barycentric_interpolate(q: [Quality; 4], w: [Real; 4]) -> Quality {
    let [q0, q1, q2, q3] = q;
    let [w0, w1, w2, w3] = w;

    let rho = w0 * q0.rho + w1 * q1.rho + w2 * q2.rho + w3 * q3.rho;
    let vp = w0 * q0.vp + w1 * q1.vp + w2 * q2.vp + w3 * q3.vp;
    let vs = w0 * q0.vs + w1 * q1.vs + w2 * q2.vs + w3 * q3.vs;

    let qp =
        (w0 * q0.qp.recip() + w1 * q1.qp.recip() + w2 * q2.qp.recip() + w3 * q3.qp.recip()).recip();
    let qs =
        (w0 * q0.qs.recip() + w1 * q1.qs.recip() + w2 * q2.qs.recip() + w3 * q3.qs.recip()).recip();

    let alpha = w0 * q0.alpha + w1 * q1.alpha + w2 * q2.alpha + w3 * q3.alpha;

    Quality {
        rho,
        vp,
        vs,
        qp,
        qs,
        alpha,
    }
}

impl Quality {
    /// Build an owned `(N, 6)` row-major float array from a slice of qualities.
    ///
    /// Column order: `[rho, vp, vs, qp, qs, alpha]`.
    pub fn from_slice(qualities: &[Quality]) -> Array2<Real> {
        let n = qualities.len();
        let flat: Vec<Real> = qualities
            .iter()
            .flat_map(|q| [q.rho, q.vp, q.vs, q.qp, q.qs, q.alpha])
            .collect();
        Array2::from_shape_vec((n, 6), flat).expect("internal error: quality array shape mismatch")
    }
}

impl From<Quality> for Array1<Real> {
    /// Convert a `Quality` to a 5-element array of `[rho, vp, vs, qp, qs]`.
    ///
    /// **`alpha` is intentionally excluded** from this array.  This conversion
    /// was established before `alpha` became a first-class field and is
    /// retained for callers that only need the five physical-property
    /// components.  Use [`Quality::from_slice`] when you need all six
    /// components (including `alpha`) as an ndarray.
    fn from(val: Quality) -> Self {
        Array1::from_iter([val.rho, val.vp, val.vs, val.qp, val.qs])
    }
}

impl From<ArrayView1<'_, Real>> for Quality {
    fn from(arr: ArrayView1<'_, Real>) -> Self {
        Quality {
            rho: arr[0],
            vp: arr[1],
            vs: arr[2],
            qp: arr[3],
            qs: arr[4],
            alpha: arr[5],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_blend_identity_alpha_one() {
        let a = Quality {
            rho: 10.0,
            vp: 20.0,
            vs: 30.0,
            qp: 40.0,
            qs: 50.0,
            alpha: 1.0,
        };
        let b = Quality {
            rho: 99.0,
            vp: 99.0,
            vs: 99.0,
            qp: 99.0,
            qs: 99.0,
            alpha: 0.5,
        };
        let blended = a.blend(&b);
        assert_relative_eq!(blended.rho, a.rho, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_commutative_alpha() {
        // Alpha composition is always commutative regardless of alpha values
        let a = Quality {
            rho: 2700.0,
            vp: 6000.0,
            vs: 3500.0,
            qp: 200.0,
            qs: 100.0,
            alpha: 0.6,
        };
        let b = Quality {
            rho: 1000.0,
            vp: 1500.0,
            vs: 0.0,
            qp: 50.0,
            qs: 25.0,
            alpha: 0.4,
        };
        let ab = a.blend(&b);
        let ba = b.blend(&a);
        assert_relative_eq!(ab.alpha, ba.alpha, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_non_commutative_materials() {
        // With different materials and alphas, blend order changes result
        let a = Quality {
            rho: 2700.0,
            vp: 6000.0,
            vs: 3500.0,
            qp: 200.0,
            qs: 100.0,
            alpha: 0.6,
        };
        let b = Quality {
            rho: 1000.0,
            vp: 1500.0,
            vs: 0.0,
            qp: 50.0,
            qs: 25.0,
            alpha: 0.4,
        };
        let ab = a.blend(&b);
        let ba = b.blend(&a);
        // Alpha is commutative, materials are not
        assert_relative_eq!(ab.alpha, ba.alpha, epsilon = 1e-5);
        assert!(ab.rho != ba.rho);
        assert!(ab.vp != ba.vp);
    }

    #[test]
    fn test_blend_epsilon_shortcut_transparent_self() {
        let a = Quality {
            rho: 1.0,
            vp: 1.0,
            vs: 1.0,
            qp: 0.0,
            qs: 1.0,
            alpha: 0.0,
        };
        let b = Quality {
            rho: 99.0,
            vp: 99.0,
            vs: 99.0,
            qp: 99.0,
            qs: 99.0,
            alpha: 0.5,
        };
        let blended = a.blend(&b);
        assert_relative_eq!(blended.rho, b.rho, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, b.alpha, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_epsilon_shortcut_transparent_rhs() {
        let a = Quality {
            rho: 10.0,
            vp: 10.0,
            vs: 10.0,
            qp: 10.0,
            qs: 10.0,
            alpha: 0.5,
        };
        let b = Quality {
            rho: 99.0,
            vp: 99.0,
            vs: 99.0,
            qp: 0.0,
            qs: 99.0,
            alpha: 0.0,
        };
        let blended = a.blend(&b);
        assert_relative_eq!(blended.rho, a.rho, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, a.alpha, epsilon = 1e-5);
    }

    #[test]
    fn test_blend_harmonic_mean_qp() {
        // alpha_a=0.5, alpha_b=0.5 → a0=2/3, a1=1/3, qp 100/300 → harmonic 900/7
        // Arithmetic mean would give 500/3 ≈ 166.667
        let a = Quality {
            rho: 1.0,
            vp: 1.0,
            vs: 1.0,
            qp: 100.0,
            qs: 1.0,
            alpha: 0.5,
        };
        let b = Quality {
            rho: 1.0,
            vp: 1.0,
            vs: 1.0,
            qp: 300.0,
            qs: 1.0,
            alpha: 0.5,
        };
        let blended = a.blend(&b);
        assert_relative_eq!(blended.qp, 900.0 / 7.0, epsilon = 1e-3);
    }

    #[test]
    fn test_blend_two_equal_half_alpha() {
        let q1 = Quality {
            rho: 10.0,
            vp: 20.0,
            vs: 30.0,
            qp: 40.0,
            qs: 50.0,
            alpha: 0.5,
        };
        let q2 = Quality {
            rho: 10.0,
            vp: 20.0,
            vs: 30.0,
            qp: 40.0,
            qs: 50.0,
            alpha: 0.5,
        };
        let blended = q1.blend(&q2);
        assert_relative_eq!(blended.rho, 10.0, epsilon = 1e-5);
        assert_relative_eq!(blended.vp, 20.0, epsilon = 1e-5);
        assert_relative_eq!(blended.alpha, 0.75, epsilon = 1e-5);
    }

    #[test]
    fn test_from_array1() {
        let arr = ndarray::array![1.0, 2.0, 3.0, 4.0, 5.0, 0.8];
        let q = Quality::from(arr.view());
        assert_relative_eq!(q.rho, 1.0);
        assert_relative_eq!(q.vp, 2.0);
        assert_relative_eq!(q.vs, 3.0);
        assert_relative_eq!(q.qp, 4.0);
        assert_relative_eq!(q.qs, 5.0);
        assert_relative_eq!(q.alpha, 0.8);
    }

    #[test]
    fn test_into_array1() {
        let quality = Quality {
            rho: 1.0,
            vp: 2.0,
            vs: 3.0,
            qp: 4.0,
            qs: 5.0,
            alpha: 0.0,
        };
        let arr: ndarray::Array1<Real> = quality.into();
        assert_eq!(arr.len(), 5);
        assert_relative_eq!(arr[0], 1.0);
        assert_relative_eq!(arr[4], 5.0);
    }

    // -----------------------------------------------------------------------
    // Property tests
    // -----------------------------------------------------------------------

    use proptest::prelude::*;

    /// Material magnitudes spanning the range the model actually carries
    /// (densities ~2700, velocities ~6000, Q ~100).  Strictly positive because
    /// `blend` takes the reciprocal of `qp`/`qs`.
    fn material() -> impl Strategy<Value = Real> {
        (1.0 as Real)..10_000.0
    }

    fn alpha() -> impl Strategy<Value = Real> {
        (0.0 as Real)..=1.0
    }

    fn quality() -> impl Strategy<Value = Quality> {
        (
            material(),
            material(),
            material(),
            material(),
            material(),
            alpha(),
        )
            .prop_map(|(rho, vp, vs, qp, qs, alpha)| Quality {
                rho,
                vp,
                vs,
                qp,
                qs,
                alpha,
            })
    }

    /// Tolerance scaled to the magnitudes involved.
    ///
    /// Near `alpha ≈ f32::EPSILON` the weights `a0 + a1` differ from 1 by ~1e-7,
    /// which at material magnitude 1e4 is ~1e-3 of overshoot past the convex
    /// hull.  A fixed tolerance would be marginally flaky.
    fn tol(magnitude: Real) -> Real {
        1e-3 + 1e-4 * magnitude
    }

    proptest! {
        /// The composited opacity is symmetric in its operands, always in
        /// `[0, 1]`, and never less than either input.
        #[test]
        fn prop_blend_alpha_is_commutative_and_monotone(a in quality(), b in quality()) {
            let ab = a.blend(&b);
            let ba = b.blend(&a);

            prop_assert!((ab.alpha - ba.alpha).abs() < 1e-5,
                         "alpha not commutative: {} vs {}", ab.alpha, ba.alpha);
            prop_assert!((0.0..=1.0 + 1e-5).contains(&ab.alpha), "alpha {} out of range", ab.alpha);
            prop_assert!(ab.alpha >= a.alpha - 1e-5, "alpha decreased: {} < {}", ab.alpha, a.alpha);
            prop_assert!(ab.alpha >= b.alpha - 1e-5, "alpha decreased: {} < {}", ab.alpha, b.alpha);
        }

        /// Every blended material lies within the convex hull of its two
        /// inputs — a weighted mean cannot leave the interval it averages over.
        ///
        /// This is the invariant that fails if a weight is negative, if `a0`
        /// and `a1` stop summing to 1, or if a field is crossed with another.
        #[test]
        fn prop_blend_materials_lie_in_convex_hull(a in quality(), b in quality()) {
            let ab = a.blend(&b);

            for (got, lo, hi) in [
                (ab.rho, a.rho.min(b.rho), a.rho.max(b.rho)),
                (ab.vp, a.vp.min(b.vp), a.vp.max(b.vp)),
                (ab.vs, a.vs.min(b.vs), a.vs.max(b.vs)),
                (ab.qp, a.qp.min(b.qp), a.qp.max(b.qp)),
                (ab.qs, a.qs.min(b.qs), a.qs.max(b.qs)),
            ] {
                let t = tol(hi);
                prop_assert!(got >= lo - t && got <= hi + t,
                             "{got} outside [{lo}, {hi}] (tol {t})");
            }
        }

        /// No blend of finite, positive inputs may produce a NaN or an
        /// infinity — the reason `blend` short-circuits on near-zero alpha in
        /// the first place.
        #[test]
        fn prop_blend_is_finite(a in quality(), b in quality()) {
            let ab = a.blend(&b);
            for (name, v) in [("rho", ab.rho), ("vp", ab.vp), ("vs", ab.vs),
                              ("qp", ab.qp), ("qs", ab.qs), ("alpha", ab.alpha)] {
                prop_assert!(v.is_finite(), "{name} is not finite: {v}");
            }
        }

        /// Blending anything under a fully-opaque quality is the identity.
        #[test]
        fn prop_blend_opaque_is_left_identity(a in quality(), b in quality()) {
            let opaque = Quality { alpha: 1.0, ..a };
            let ab = opaque.blend(&b);
            prop_assert!((ab.rho - opaque.rho).abs() < tol(opaque.rho));
            prop_assert!((ab.vp - opaque.vp).abs() < tol(opaque.vp));
            prop_assert!((ab.alpha - 1.0).abs() < 1e-5);
        }

        /// `qp`/`qs` are combined as a *harmonic* mean, which for distinct
        /// inputs is **strictly** below the arithmetic mean of the same values.
        ///
        /// The strictness is the point: asserting only `<=` would be satisfied
        /// by a plain weighted sum, so swapping the `recip()` chain for
        /// arithmetic averaging would go unnoticed here.  The preconditions
        /// keep the AM-HM gap comfortably above the f32 noise floor.
        #[test]
        fn prop_blend_q_is_strictly_harmonic(a in quality(), b in quality()) {
            let alpha = a.alpha + b.alpha * (1.0 - a.alpha);
            prop_assume!(alpha > 1e-3);
            let a0 = a.alpha / alpha;
            let a1 = b.alpha * (1.0 - a.alpha) / alpha;

            // Both operands must genuinely participate...
            prop_assume!(a0 > 0.1 && a1 > 0.1);
            // ...and the values must differ by at least 2x, which bounds the
            // AM-HM gap below at ~4% of the arithmetic mean.
            let (lo, hi) = (a.qp.min(b.qp), a.qp.max(b.qp));
            prop_assume!(hi >= 2.0 * lo);

            let ab = a.blend(&b);
            let arithmetic = a0 * a.qp + a1 * b.qp;

            prop_assert!(ab.qp < arithmetic * 0.99,
                         "qp {} is not strictly below the arithmetic mean {}",
                         ab.qp, arithmetic);
            // Still a mean: bounded by the inputs it averages.
            prop_assert!(ab.qp >= lo - tol(hi) && ab.qp <= hi + tol(hi));
        }
    }
}
