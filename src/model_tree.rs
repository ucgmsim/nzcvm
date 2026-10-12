use std::time::Instant;

use approx::abs_diff_eq;
use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh::bvh::{Bvh, BvhNode};
use deepsize::{Context, DeepSizeOf};
use serde::Serialize;

use crate::mesh::MeshModel;
use crate::model::AnyField;
use crate::quality::Quality;
use crate::query::{Explanation, ModelContribution, Query, QueryStats};
use crate::real::Real;
use crate::tree_query::{Contains, priority_ray_iterator, priority_ray_stats_iterator};
use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::BHShape;
use nalgebra::{Affine3, Point, Point3, Point4};

const ALPHA_SATURATED: Real = 1.0;
const ALPHA_EPS: Real = 1e-4;

/// Half-unit extent added to the priority dimension of the 4D AABB to keep
/// the AABB non-degenerate. The value 0.5 sits between any two consecutive
/// integer priorities, so BVH node AABBs correctly reflect the minimum and
/// maximum priority reachable through each subtree.
const PRIORITY_AABB_EXTENT: Real = 0.5;

/// The vertical coordinate a mesh model's vertices use, both positive down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Vertical {
    /// Negated elevation, so the model stays put as the topography changes:
    /// tomography.
    Elevation,
    /// Depth below the topography, so the model follows it: basins.
    Depth,
}

/// Serialisable summary of a [`TreeModel`] for diagnostics.
#[derive(Serialize)]
pub struct MeshModelView {
    pub id: usize,
    pub name: String,
    pub bounds: [f32; 6],
    pub transform: Option<Affine3<Real>>,
    pub priority: u8,
    pub vertical: Vertical,
    /// In-memory size of the model in bytes.
    pub size: usize,
}

/// A mesh model with what the [`ModelTree`] needs to rank it and index it.
pub struct TreeModel {
    pub model: MeshModel<AnyField>,
    /// Lower numbers take precedence.
    pub priority: u8,
    /// Human-readable name for this mesh model.
    pub name: String,
    /// Which coordinate of a query point the mesh's third axis matches.
    pub vertical: Vertical,
    /// Position in the tree, assigned by [`ModelTree::new`].
    pub id: usize,
    node_index: usize,
}

impl TreeModel {
    pub fn new(model: MeshModel<AnyField>, priority: u8, name: String, vertical: Vertical) -> Self {
        Self {
            model,
            priority,
            name,
            vertical,
            id: 0,
            node_index: 0,
        }
    }

    pub fn pretty_print(&self) {
        let name_display = if self.name.is_empty() {
            format!("(id={})", self.id)
        } else {
            format!("{:?}", self.name)
        };
        let mesh = self.model.mesh();
        println!(
            "Mesh model {} with {} vertices and {} simplices, tree depth = {}, priority = {}.",
            name_display,
            mesh.num_vertices(),
            mesh.num_simplices(),
            mesh.depth(),
            self.priority,
        )
    }

    pub fn view(&self) -> MeshModelView {
        let aabb = self.model.aabb3();
        // The view is a diagnostics summary, so `bounds` stays f32 regardless
        // of `Real`, keeping the serialised schema stable.  The casts are
        // no-ops in the default build and load-bearing under `high_precision`.
        #[allow(clippy::unnecessary_cast)]
        let bounds: [f32; 6] = [
            aabb.min.x as f32,
            aabb.min.y as f32,
            aabb.min.z as f32,
            aabb.max.x as f32,
            aabb.max.y as f32,
            aabb.max.z as f32,
        ];
        MeshModelView {
            id: self.id,
            name: self.name.clone(),
            bounds,
            transform: self.model.mesh().transform(),
            priority: self.priority,
            vertical: self.vertical,
            size: self.deep_size_of(),
        }
    }
}

impl DeepSizeOf for TreeModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.model.heap_size() + self.name.deep_size_of_children(context)
    }
}

/// `Contains` yields `(priority, quality)` when a simplex of the model
/// contains the query point `(x, y, z, depth)`, so the outer `ModelTree`
/// traversal gets the result without a second lookup. The model reads the
/// vertical coordinate its mesh uses.
impl Contains<Real, 4, (u8, Quality)> for TreeModel {
    fn contains(&self, query_point: &Point4<Real>) -> Option<(u8, Quality)> {
        let vertical = match self.vertical {
            Vertical::Elevation => query_point.z,
            Vertical::Depth => query_point.w,
        };
        let point = Point3::new(query_point.x, query_point.y, vertical);
        self.model.query(point).map(|q| (self.priority, q))
    }
}

/// 3-D AABB used by the outer `ModelTree` BVH: `(x, y, priority)`.
///
/// The tree prunes on the horizontal extent alone, since models differ in
/// which vertical coordinate they use; each model's own mesh tests the
/// vertical. The last dimension is the model's priority: both ends are set
/// to `priority` (with a half-unit extent so the AABB is non-degenerate),
/// which lets the priority-ray traversal use `aabb.min[2]` as the `t_min`
/// hit distance and thereby visit models in priority order.
impl Bounded<Real, 3> for TreeModel {
    fn aabb(&self) -> Aabb<Real, 3> {
        let aabb = self.model.aabb3();
        let p = self.priority as Real;
        let min = Point::<Real, 3>::from([aabb.min.x, aabb.min.y, p]);
        let max = Point::<Real, 3>::from([aabb.max.x, aabb.max.y, p + PRIORITY_AABB_EXTENT]);
        Aabb::with_bounds(min, max)
    }
}

impl BHShape<Real, 3> for TreeModel {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[derive(Serialize)]
pub struct ModelTreeView {
    pub models: Vec<MeshModelView>,
    pub size: usize,
}

/// A collection of [`TreeModel`]s indexed by a BVH over `(x, y, priority)`.
///
/// A ray along the priority axis visits models in ascending priority order
/// (lowest number = highest priority).
pub struct ModelTree {
    bvh_tree: Bvh<Real, 3>,
    models: Vec<TreeModel>,
}

impl DeepSizeOf for ModelTree {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // The outer BVH is 3-dimensional (x, y, priority), so the node size
        // must use `BvhNode<Real, 3>`.
        self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 3>>()
            + self.models.deep_size_of_children(context)
    }
}

impl ModelTree {
    pub fn new(mut models: Vec<TreeModel>) -> Self {
        for (i, model) in models.iter_mut().enumerate() {
            model.id = i;
        }
        let bvh_tree = Bvh::build_par(&mut models);
        Self { bvh_tree, models }
    }

    pub fn aabb(&self) -> bvh::aabb::Aabb<Real, 3> {
        self.models
            .iter()
            .map(|m| m.model.aabb3())
            .reduce(|a, b| a.join(&b))
            .expect("ModelTree must contain at least one model")
    }

    pub fn pretty_print(&self) {
        println!("ModelTree with {} mesh models:", self.models.len());
        for model in &self.models {
            model.pretty_print();
        }
    }

    pub fn view(&self) -> ModelTreeView {
        ModelTreeView {
            models: self.models.iter().map(|model| model.view()).collect(),
            size: self.deep_size_of(),
        }
    }
}

fn blend_accumulate(
    mut quality: Option<Quality>,
    iter: impl Iterator<Item = (u8, Quality)>,
) -> Option<Quality> {
    for (_, q) in iter {
        quality = Some(match quality {
            None => q,
            Some(current) => {
                if abs_diff_eq!(current.alpha, ALPHA_SATURATED, epsilon = ALPHA_EPS) {
                    return Some(current);
                }
                current.blend(&q)
            }
        });
    }
    quality
}

impl Query for ModelTree {
    type Explanation = Explanation;

    fn query(
        &self,
        point: Point4<Real>,
        existing: Option<Quality>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality> {
        blend_accumulate(
            existing,
            priority_ray_iterator(&self.bvh_tree, &self.models, point, lo as Real, hi as Real),
        )
    }

    fn query_stats(&self, point: Point4<Real>) -> QueryStats {
        let now = Instant::now();
        // NOTE: This method traverses the BVH *twice* — once for AABB/simplex
        // counts and once for the blended quality — so `elapsed` reflects two
        // full traversals (~2× a normal query).  Separating them would require
        // a single combined iterator; the current split keeps the hot paths
        // simpler and this diagnostic path is never on the critical path.
        let mut iter = priority_ray_stats_iterator(&self.bvh_tree, &self.models, point, 0.0, 255.0);
        for _ in iter.by_ref() {}
        let outer_stats = iter.stats;
        let mut hit_count = 0;
        let quality = blend_accumulate(
            None,
            priority_ray_iterator(&self.bvh_tree, &self.models, point, 0.0, 255.0)
                .inspect(|_| hit_count += 1),
        );
        QueryStats {
            aabb_tests: outer_stats.aabb_tests,
            simplex_tests: outer_stats.simplex_tests,
            hit_count,
            output: quality,
            elapsed: now.elapsed().as_nanos() as u64,
        }
    }

    fn query_explain(&self, point: Point4<Real>) -> Explanation {
        let mut contributions = Vec::new();
        let mut blended: Option<Quality> = None;
        let mut termination = None;
        for (i, (priority, q)) in
            priority_ray_iterator(&self.bvh_tree, &self.models, point, 0.0, 255.0).enumerate()
        {
            contributions.push(ModelContribution {
                priority,
                quality: q,
            });
            blended = Some(match blended {
                None => q,
                Some(ref current) => {
                    if abs_diff_eq!(current.alpha, ALPHA_SATURATED, epsilon = ALPHA_EPS)
                        && termination.is_none()
                    {
                        termination = Some(i);
                    }
                    // Intentionally continue blending past saturation so that
                    // `contributions` is a complete record of every model that
                    // was visited, not just those that contributed before
                    // saturation.  The `termination` index marks the point at
                    // which further blending became a no-op.
                    current.blend(&q)
                }
            });
        }
        Explanation {
            contributions,
            output: blended,
            termination,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Point3, Point4};

    use crate::mesh::MeshModel;
    use crate::model::{Constant, Interpolated};
    use crate::quality::Quality;
    use crate::real::Real;

    fn cube_mesh(priority: u8, val: Real, alpha: Real) -> TreeModel {
        vertical_cube_mesh(priority, val, alpha, Vertical::Elevation)
    }

    fn vertical_cube_mesh(priority: u8, val: Real, alpha: Real, vertical: Vertical) -> TreeModel {
        let v: Vec<Point3<Real>> = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 1.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let faces = vec![
            [0, 1, 2, 4],
            [3, 1, 2, 7],
            [5, 1, 4, 7],
            [6, 2, 4, 7],
            [1, 2, 4, 7],
        ];
        let q = Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
            alpha,
        };
        let field = Interpolated::new(vec![q; v.len()]);
        let model = MeshModel::new(v, faces, field, None).unwrap();
        TreeModel::new(model.into_any(), priority, String::new(), vertical)
    }

    /// Inside the unit cube in both elevation and depth.
    const PT: Point4<Real> = Point4::new(0.5, 0.5, 0.5, 0.5);

    #[test]
    fn test_query_hit_and_miss() {
        let tree = ModelTree::new(vec![cube_mesh(0, 5.0, 1.0)]);
        assert_relative_eq!(
            tree.query(PT, None, 0, 255).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        assert!(
            tree.query(Point4::new(10.0, 10.0, 10.0, 10.0), None, 0, 255)
                .is_none()
        );
    }

    #[test]
    fn test_priority_ordering() {
        // Lower priority number wins regardless of insertion order.
        let tree = ModelTree::new(vec![cube_mesh(1, 10.0, 1.0), cube_mesh(0, 5.0, 1.0)]);
        assert_relative_eq!(
            tree.query(PT, None, 0, 255).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_alpha_blending() {
        let tree = ModelTree::new(vec![cube_mesh(0, 0.0, 0.5), cube_mesh(1, 10.0, 1.0)]);
        let q = tree.query(PT, None, 0, 255).unwrap();
        assert_relative_eq!(q.rho, 5.0, epsilon = 1e-4);
        assert_relative_eq!(q.alpha, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_query_bounded() {
        let tree = ModelTree::new(vec![cube_mesh(10, 5.0, 1.0), cube_mesh(200, 99.0, 1.0)]);
        assert_relative_eq!(
            tree.query(PT, None, 0, 127).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            tree.query(PT, None, 128, 255).unwrap().rho,
            99.0,
            epsilon = 1e-4
        );
    }

    /// The `(lo, hi)` priority window is inclusive at both ends.
    ///
    /// A half-open window would exclude the `hi == priority` model, and an
    /// exclusive lower bound would exclude the `lo == priority` one.
    #[test]
    fn test_query_priority_bounds_are_inclusive() {
        let tree = ModelTree::new(vec![cube_mesh(10, 5.0, 1.0)]);

        // Degenerate window on the model's own priority.
        assert_relative_eq!(
            tree.query(PT, None, 10, 10).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        // lo == priority
        assert_relative_eq!(
            tree.query(PT, None, 10, 20).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        // hi == priority
        assert_relative_eq!(
            tree.query(PT, None, 0, 10).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );

        // Just outside on either side must miss entirely.
        assert!(tree.query(PT, None, 11, 20).is_none());
        assert!(tree.query(PT, None, 0, 9).is_none());
    }

    /// Adjacent priorities must be discriminated.
    ///
    /// This is what exercises `PRIORITY_AABB_EXTENT = 0.5` (`mesh.rs`): the
    /// half-unit padding on the priority dimension of the AABB has to sit
    /// strictly between two consecutive integers, or a query for priority 10
    /// would also sweep in the priority-11 model.
    #[test]
    fn test_adjacent_priorities_are_discriminated() {
        let tree = ModelTree::new(vec![cube_mesh(10, 5.0, 1.0), cube_mesh(11, 99.0, 1.0)]);

        assert_relative_eq!(
            tree.query(PT, None, 10, 10).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        assert_relative_eq!(
            tree.query(PT, None, 11, 11).unwrap().rho,
            99.0,
            epsilon = 1e-4
        );
        // Spanning both, the lower priority number wins.
        assert_relative_eq!(
            tree.query(PT, None, 10, 11).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_blend_into_existing() {
        let tree = ModelTree::new(vec![cube_mesh(0, 4.0, 0.5)]);
        let existing = Some(Quality {
            rho: 10.0,
            vp: 10.0,
            vs: 10.0,
            qp: 10.0,
            qs: 10.0,
            alpha: 0.5,
        });
        let q = tree.query(PT, existing, 0, 255).unwrap();
        // alpha = 0.5 + 0.5*(1-0.5) = 0.75; rho = (0.5/0.75)*10 + (0.25/0.75)*4
        assert_relative_eq!(q.alpha, 0.75, epsilon = 1e-4);
        assert_relative_eq!(
            q.rho,
            (0.5 / 0.75) * 10.0 + (0.25 / 0.75) * 4.0,
            epsilon = 1e-4
        );
    }

    #[test]
    fn test_query_stats() {
        let tree = ModelTree::new(vec![cube_mesh(0, 5.0, 1.0)]);
        let stats = tree.query_stats(PT);
        assert!(stats.hit_count >= 1);
        assert!(stats.output.is_some());
    }

    #[test]
    fn test_query_explain() {
        let tree = ModelTree::new(vec![cube_mesh(0, 5.0, 0.5), cube_mesh(1, 10.0, 1.0)]);
        let e = tree.query_explain(PT);
        assert!(e.contributions.len() >= 2);
        assert_eq!(e.contributions[0].priority, 0);
    }

    #[test]
    fn test_constant_model() {
        let v = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let faces = vec![[0, 1, 2, 3]];
        let q = Quality {
            rho: 42.0,
            vp: 1.0,
            vs: 1.0,
            qp: 1.0,
            qs: 1.0,
            alpha: 1.0,
        };
        let model = MeshModel::new(v, faces, Constant::new(vec![q]), None).unwrap();
        let tree = ModelTree::new(vec![TreeModel::new(
            model.into_any(),
            0,
            String::new(),
            Vertical::Elevation,
        )]);
        assert_relative_eq!(
            tree.query(Point4::new(0.1, 0.1, 0.1, 0.1), None, 0, 255)
                .unwrap()
                .rho,
            42.0,
            epsilon = 1e-4
        );
    }

    /// Each model reads the vertical coordinate its mesh uses: an elevation
    /// model the query's z, a depth model its depth.
    #[test]
    fn test_models_read_their_own_vertical_coordinate() {
        let tree = ModelTree::new(vec![
            vertical_cube_mesh(0, 5.0, 1.0, Vertical::Depth),
            vertical_cube_mesh(1, 9.0, 1.0, Vertical::Elevation),
        ]);
        // Inside both: the depth model has the higher priority.
        assert_relative_eq!(
            tree.query(Point4::new(0.5, 0.5, 0.5, 0.5), None, 0, 255)
                .unwrap()
                .rho,
            5.0,
            epsilon = 1e-4
        );
        // Deeper than the depth model reaches, but inside the elevation one.
        assert_relative_eq!(
            tree.query(Point4::new(0.5, 0.5, 0.5, 5.0), None, 0, 255)
                .unwrap()
                .rho,
            9.0,
            epsilon = 1e-4
        );
        // Inside the depth model only, by depth, whatever the elevation.
        assert_relative_eq!(
            tree.query(Point4::new(0.5, 0.5, 50.0, 0.5), None, 0, 255)
                .unwrap()
                .rho,
            5.0,
            epsilon = 1e-4
        );
        assert!(
            tree.query(Point4::new(0.5, 0.5, 50.0, 5.0), None, 0, 255)
                .is_none()
        );
    }
}
