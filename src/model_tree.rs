use std::time::Instant;

use approx::abs_diff_eq;
use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh::bvh::{Bvh, BvhNode};
use deepsize::{Context, DeepSizeOf};
use serde::Serialize;

use crate::mesh::{MeshModel, MeshModelView};
use crate::quality::Quality;
use crate::query::{Explanation, ModelContribution, Query, QueryStats};
use crate::real::Real;
use crate::tree_query::{priority_ray_iterator, priority_ray_stats_iterator};
use nalgebra::Point3;

const ALPHA_SATURATED: Real = 1.0;
const ALPHA_EPS: Real = 1e-4;

#[derive(Serialize)]
pub struct ModelTreeView {
    pub models: Vec<MeshModelView>,
    pub size: usize,
}

/// A collection of [`MeshModel`]s indexed by a 4-D BVH.
///
/// The fourth dimension encodes model priority; a ray along that axis visits
/// models in ascending priority order (lowest number = highest priority).
pub struct ModelTree {
    bvh_tree: Bvh<Real, 4>,
    models: Vec<MeshModel>,
}

impl DeepSizeOf for ModelTree {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        // The outer BVH is 4-dimensional (3 spatial + 1 priority), so the node
        // size must use `BvhNode<Real, 4>` to account for the extra AABB extent.
        self.bvh_tree.nodes.capacity() * size_of::<BvhNode<Real, 4>>()
            + self.models.deep_size_of_children(context)
    }
}

impl ModelTree {
    pub fn new(mut models: Vec<MeshModel>) -> Self {
        for (i, model) in models.iter_mut().enumerate() {
            model.id = i;
        }
        let bvh_tree = Bvh::build_par(&mut models);
        Self { bvh_tree, models }
    }

    pub fn aabb(&self) -> bvh::aabb::Aabb<Real, 3> {
        self.models
            .iter()
            .map(|m| m.aabb3())
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
        point: Point3<Real>,
        existing: Option<Quality>,
        lo: u8,
        hi: u8,
    ) -> Option<Quality> {
        blend_accumulate(
            existing,
            priority_ray_iterator(&self.bvh_tree, &self.models, point, lo as Real, hi as Real),
        )
    }

    fn query_stats(&self, point: Point3<Real>) -> QueryStats {
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

    fn query_explain(&self, point: Point3<Real>) -> Explanation {
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
    use crate::model::{ConstantModel, InterpolateModel, Model};
    use crate::quality::Quality;
    use crate::real::Real;

    fn cube_mesh(priority: u8, val: Real, alpha: Real) -> MeshModel {
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
            Point4::new(0usize, 1, 2, 4),
            Point4::new(3, 1, 2, 7),
            Point4::new(5, 1, 4, 7),
            Point4::new(6, 2, 4, 7),
            Point4::new(1, 2, 4, 7),
        ];
        let q = Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
            alpha,
        };
        let qualities = vec![q; v.len()];
        let models = faces
            .iter()
            .map(|f| {
                Model::from(InterpolateModel {
                    qualities: f.map(|i| i as u32),
                })
            })
            .collect();
        MeshModel::new(v, faces, models, qualities, priority, None, String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"))
    }

    const PT: Point3<Real> = Point3::new(0.5, 0.5, 0.5);

    #[test]
    fn test_query_hit_and_miss() {
        let tree = ModelTree::new(vec![cube_mesh(0, 5.0, 1.0)]);
        assert_relative_eq!(
            tree.query(PT, None, 0, 255).unwrap().rho,
            5.0,
            epsilon = 1e-4
        );
        assert!(
            tree.query(Point3::new(10.0, 10.0, 10.0), None, 0, 255)
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
    /// half-unit padding on the priority dimension of the 4D AABB has to sit
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
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let q = Quality {
            rho: 42.0,
            vp: 1.0,
            vs: 1.0,
            qp: 1.0,
            qs: 1.0,
            alpha: 1.0,
        };
        let models = vec![Model::from(ConstantModel { quality: 0 })];
        let mesh = MeshModel::new(v, faces, models, vec![q], 0, None, String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        let tree = ModelTree::new(vec![mesh]);
        assert_relative_eq!(
            tree.query(Point3::new(0.1, 0.1, 0.1), None, 0, 255)
                .unwrap()
                .rho,
            42.0,
            epsilon = 1e-4
        );
    }
}
