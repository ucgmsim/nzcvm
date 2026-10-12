use std::cmp::Ordering;

use bvh::{
    aabb::{Aabb, Bounded},
    bounding_hierarchy::BHValue,
    bvh::{Bvh, BvhNode},
};
use nalgebra::Point;
use small_collections::SmallBinaryHeap;
use smallvec::SmallVec;

use crate::real::Real;

/// A shape that can test whether it contains a query point.
///
/// `Data` is the value produced when the shape *does* contain the point.
/// Returning `Option<Data>` instead of `bool` lets the caller reuse the
/// computation that was performed during the containment test (e.g. the
/// queried quality of a mesh model) without having to repeat it.
pub trait Contains<T: BHValue, const D: usize, Data> {
    fn contains(&self, point: &Point<T, D>) -> Option<Data>;
}

const SMALL_VEC_RAY_SIZE: usize = 8;
/// Stack capacity for the stats iterator.  Larger than `SMALL_VEC_RAY_SIZE`
/// because diagnostic traversals visit every matching leaf (no early-exit on
/// alpha saturation) so the stack may grow deeper.
const SMALL_VEC_STATS_SIZE: usize = 16;

#[derive(Debug, Clone, Copy, Default)]
pub struct TraversalStats {
    pub aabb_tests: usize,
    pub simplex_tests: usize,
}

// ---------------------------------------------------------------------------
// Priority-ray traversal
// ---------------------------------------------------------------------------

/// Tests whether a query point lies within the first `DBVH - 1` dimensions
/// of a `DBVH`-dimensional AABB, and whether the subtree's priority range
/// overlaps `[priority_lo, priority_hi]`.  The point may carry more than
/// `DBVH - 1` coordinates: the AABB tests only the leading ones, and the
/// shapes' containment tests see them all.
///
/// Returns `Some(aabb.min[DBVH - 1])` — the minimum value along the last
/// dimension (i.e. the minimum priority reachable through this subtree) —
/// on hit, or `None` on a spatial miss or priority-range miss.
///
/// This implements the AABB intersection test for the conceptual ray
/// `origin = (p[0], …, p[DBVH-2], 0)`, `direction = (0, …, 0, 1)`.
///
/// For a leaf at priority `p` the AABB's priority dimension is `[p, p + 0.5]`.
/// For an internal node it spans `[min_priority_in_subtree, max_priority_in_subtree + 0.5]`.
/// The subtree is pruned when:
/// - its minimum priority exceeds `priority_hi`, or
/// - its maximum priority (= `aabb.max[DBVH - 1] - 0.5`) is less than
///   `priority_lo` (equivalently `aabb.max[DBVH - 1] <= priority_lo`, since
///   priorities are integers and the extent is exactly `0.5`).
#[inline]
fn priority_aabb_hit_bounded<T: BHValue, const DBVH: usize, const DQ: usize>(
    point: &Point<T, DQ>,
    aabb: &Aabb<T, DBVH>,
    priority_lo: T,
    priority_hi: T,
) -> Option<T> {
    let priority_axis = DBVH - 1;
    for i in 0..priority_axis {
        if point.coords[i] < aabb.min.coords[i] || point.coords[i] > aabb.max.coords[i] {
            return None;
        }
    }
    let p_min = aabb.min.coords[priority_axis];
    let p_max = aabb.max.coords[priority_axis];
    if p_min > priority_hi || p_max <= priority_lo {
        return None;
    }
    Some(p_min)
}

#[derive(PartialEq)]
struct Priority(Real, usize);

impl Eq for Priority {}

impl PartialOrd for Priority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Priority {
    fn cmp(&self, other: &Self) -> Ordering {
        let Priority(p, i) = self;
        let Priority(p_o, i_o) = other;
        // The .reverse() call ensures that priority 0 > priority 255
        p.total_cmp(p_o).reverse().then(i.cmp(i_o))
    }
}

/// Iterator over a `DBVH`-dimensional BVH that models a ray from
/// `(point, 0)` along the `DBVH`-th axis (the *priority* dimension).
///
/// Leaves are visited in ascending order of `aabb.min[DBVH - 1]` (= priority),
/// so callers receive results from the highest-priority model first.
/// At each internal node the child with the lower `t_min` (= lower minimum
/// priority value in that subtree) is pushed on top of the stack and
/// therefore visited first.
///
/// The optional priority bounds `[priority_lo, priority_hi]` restrict which
/// leaves are visited: subtrees whose priority range does not overlap
/// `[priority_lo, priority_hi]` are pruned, and individual leaves are
/// skipped when their own priority is out of range.
///
/// For each leaf the iterator calls `shape.contains(point_DQ)` and yields
/// the returned `Data` if the shape contains the point.
pub struct PriorityRayIterator<
    'bvh,
    'shape,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<Real, DQ, Data> + Bounded<Real, DBVH>,
> {
    bvh: &'bvh Bvh<Real, DBVH>,
    point: Point<Real, DQ>,
    shapes: &'shape [Shape],
    stack: SmallBinaryHeap<Priority, SMALL_VEC_RAY_SIZE>,
    priority_lo: Real,
    priority_hi: Real,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, const DBVH: usize, const DQ: usize, Data, Shape>
    PriorityRayIterator<'bvh, 'shape, DBVH, DQ, Data, Shape>
where
    Shape: Bounded<Real, DBVH> + Contains<Real, DQ, Data>,
{
    fn new(
        bvh: &'bvh Bvh<Real, DBVH>,
        point: Point<Real, DQ>,
        shapes: &'shape [Shape],
        priority_lo: Real,
        priority_hi: Real,
    ) -> Self {
        let mut stack = SmallBinaryHeap::new();
        if !shapes.is_empty() {
            stack.push(Priority(0.0, 0));
        }
        Self {
            bvh,
            point,
            shapes,
            stack,
            priority_lo,
            priority_hi,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, const DBVH: usize, const DQ: usize, Data, Shape> Iterator
    for PriorityRayIterator<'_, 'shape, DBVH, DQ, Data, Shape>
where
    Shape: Bounded<Real, DBVH> + Contains<Real, DQ, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(Priority(_, node_idx)) = self.stack.pop() {
            match self.bvh.nodes[node_idx] {
                BvhNode::Leaf { shape_index, .. } => {
                    let shape = &self.shapes[shape_index];
                    // Check this leaf's own priority is within bounds before
                    // running the (potentially expensive) containment test.
                    let leaf_p = shape.aabb().min.coords[DBVH - 1];
                    if leaf_p < self.priority_lo || leaf_p > self.priority_hi {
                        continue;
                    }
                    if let Some(data) = shape.contains(&self.point) {
                        return Some(data);
                    }
                }
                BvhNode::Node {
                    child_l_index,
                    ref child_l_aabb,
                    child_r_index,
                    ref child_r_aabb,
                    ..
                } => {
                    let l_t = priority_aabb_hit_bounded(
                        &self.point,
                        child_l_aabb,
                        self.priority_lo,
                        self.priority_hi,
                    );
                    let r_t = priority_aabb_hit_bounded(
                        &self.point,
                        child_r_aabb,
                        self.priority_lo,
                        self.priority_hi,
                    );

                    match (l_t, r_t) {
                        (None, None) => {}
                        (Some(lp), None) => self.stack.push(Priority(lp, child_l_index)),
                        (None, Some(rp)) => self.stack.push(Priority(rp, child_r_index)),
                        (Some(lp), Some(rp)) => {
                            self.stack.push(Priority(lp, child_l_index));
                            self.stack.push(Priority(rp, child_r_index));
                        }
                    }
                }
            }
        }
        None
    }
}

/// Return an iterator over all models whose priority lies in `[priority_lo, priority_hi]`
/// (inclusive), visited in ascending priority order.
pub fn priority_ray_iterator<
    'bvh,
    'shape,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<Real, DQ, Data> + Bounded<Real, DBVH>,
>(
    bvh_tree: &'bvh Bvh<Real, DBVH>,
    shapes: &'shape [Shape],
    point: Point<Real, DQ>,
    priority_lo: Real,
    priority_hi: Real,
) -> PriorityRayIterator<'bvh, 'shape, DBVH, DQ, Data, Shape> {
    PriorityRayIterator::new(bvh_tree, point, shapes, priority_lo, priority_hi)
}

/// Like [`PriorityRayIterator`] but also tracks AABB and leaf-test counts.
pub struct PriorityRayStatsIterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
> {
    bvh: &'bvh Bvh<T, DBVH>,
    point: Point<T, DQ>,
    shapes: &'shape [Shape],
    stack: SmallVec<[usize; SMALL_VEC_STATS_SIZE]>,
    priority_lo: T,
    priority_hi: T,
    pub stats: TraversalStats,
    _phantom: std::marker::PhantomData<Data>,
}

impl<'bvh, 'shape, T, const DBVH: usize, const DQ: usize, Data, Shape>
    PriorityRayStatsIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    fn new(
        bvh: &'bvh Bvh<T, DBVH>,
        point: Point<T, DQ>,
        shapes: &'shape [Shape],
        priority_lo: T,
        priority_hi: T,
    ) -> Self {
        let mut stack = SmallVec::new();
        if !shapes.is_empty() {
            stack.push(0usize);
        }
        Self {
            bvh,
            point,
            shapes,
            stack,
            priority_lo,
            priority_hi,
            stats: TraversalStats::default(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'shape, T, const DBVH: usize, const DQ: usize, Data, Shape> Iterator
    for PriorityRayStatsIterator<'_, 'shape, T, DBVH, DQ, Data, Shape>
where
    T: BHValue,
    Shape: Bounded<T, DBVH> + Contains<T, DQ, Data>,
{
    type Item = Data;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node_idx) = self.stack.pop() {
            match self.bvh.nodes[node_idx] {
                BvhNode::Leaf { shape_index, .. } => {
                    let shape = &self.shapes[shape_index];
                    let leaf_p = shape.aabb().min.coords[DBVH - 1];
                    if leaf_p < self.priority_lo || leaf_p > self.priority_hi {
                        continue;
                    }
                    self.stats.simplex_tests += 1;
                    if let Some(data) = shape.contains(&self.point) {
                        return Some(data);
                    }
                }
                BvhNode::Node {
                    child_l_index,
                    ref child_l_aabb,
                    child_r_index,
                    ref child_r_aabb,
                    ..
                } => {
                    self.stats.aabb_tests += 2;

                    let l_t = priority_aabb_hit_bounded(
                        &self.point,
                        child_l_aabb,
                        self.priority_lo,
                        self.priority_hi,
                    );
                    let r_t = priority_aabb_hit_bounded(
                        &self.point,
                        child_r_aabb,
                        self.priority_lo,
                        self.priority_hi,
                    );

                    match (l_t, r_t) {
                        (None, None) => {}
                        (Some(_), None) => self.stack.push(child_l_index),
                        (None, Some(_)) => self.stack.push(child_r_index),
                        (Some(lt), Some(rt)) => {
                            if lt <= rt {
                                self.stack.push(child_r_index);
                                self.stack.push(child_l_index);
                            } else {
                                self.stack.push(child_l_index);
                                self.stack.push(child_r_index);
                            }
                        }
                    }
                }
            }
        }
        None
    }
}

pub fn priority_ray_stats_iterator<
    'bvh,
    'shape,
    T: BHValue,
    const DBVH: usize,
    const DQ: usize,
    Data,
    Shape: Contains<T, DQ, Data> + Bounded<T, DBVH>,
>(
    bvh_tree: &'bvh Bvh<T, DBVH>,
    shapes: &'shape [Shape],
    point: Point<T, DQ>,
    priority_lo: T,
    priority_hi: T,
) -> PriorityRayStatsIterator<'bvh, 'shape, T, DBVH, DQ, Data, Shape> {
    PriorityRayStatsIterator::new(bvh_tree, point, shapes, priority_lo, priority_hi)
}
