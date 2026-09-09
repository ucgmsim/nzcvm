use crate::quality::Quality;
use crate::real::Real;
use nalgebra::Point3;
use serde::Serialize;

/// Performance counters collected during a single model query.
#[derive(Debug, Serialize)]
pub struct QueryStats {
    pub aabb_tests: usize,
    pub simplex_tests: usize,
    pub hit_count: usize,
    pub output: Option<Quality>,
    /// Wall-clock time for the full query in nanoseconds.
    pub elapsed: u64,
}

/// One model's contribution to a blended query result.
#[derive(Serialize)]
pub struct ModelContribution {
    pub priority: u8,
    pub quality: Quality,
}

/// Diagnostic breakdown of a query, listing every model that contributed.
#[derive(Serialize)]
pub struct Explanation {
    pub contributions: Vec<ModelContribution>,
    pub output: Option<Quality>,
    pub termination: Option<usize>,
}

/// Spatial query interface for velocity models.
pub trait Query {
    type Explanation;

    /// Blend models whose priority falls in `[lo, hi]` into `existing` at `point`.
    ///
    /// Pass `existing = None` for a fresh query and `lo = 0, hi = 255` to
    /// consider all models.  This single method subsumes the old
    /// `query`, `query_bounded`, `query_into`, and `query_bounded_into`.
    fn query(
        &self,
        point: Point3<Real>,
        existing: Option<Quality>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality>;
    fn query_stats(&self, point: Point3<Real>) -> QueryStats;
    fn query_explain(&self, point: Point3<Real>) -> Self::Explanation;
}
