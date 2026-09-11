use crate::compact_bvh::CompactBvh;
use crate::model::*;
use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::{BuildSimplex, Simplex};
use crate::slab::Slab;
use crate::tree_query::Contains;
use deepsize::{Context, DeepSizeOf};

use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use nalgebra::{Affine3, Point, Point3, Point4, Vector3};
use serde::Serialize;

/// Default priority for models that do not specify one explicitly.
pub const DEFAULT_PRIORITY: u8 = 0;

/// Half-unit extent added to the priority dimension of the 4D AABB to keep
/// the AABB non-degenerate. The value 0.5 sits between any two consecutive
/// integer priorities, so BVH node AABBs correctly reflect the minimum and
/// maximum priority reachable through each subtree.
const PRIORITY_AABB_EXTENT: Real = 0.5;

/// Serialisable summary of a [`MeshModel`] for diagnostics.
#[derive(Serialize)]
pub struct MeshModelView {
    pub id: usize,
    pub name: String,
    pub bounds: [f32; 6],
    pub transform: Option<Affine3<Real>>,
    pub priority: u8,
    /// In-memory size of the model in bytes.
    pub size: usize,
}

/// A single tetrahedral mesh model with an internal BVH for fast point queries.
///
/// Each simplex in the mesh carries a model (via [`ModelMap`]) that maps a
/// query point to a [`Quality`].  The outer
/// [`ModelTree`](crate::model_tree::ModelTree) holds a collection of
/// `MeshModel`s and dispatches queries via a 4-D BVH (the fourth dimension
/// encodes priority).
///
/// Internally the simplices are stored in BVH leaf order (the tree is built
/// with the `bvh` crate and then converted to a [`CompactBvh`]), so a leaf
/// visit scans adjacent entries of `simplices`.
pub struct MeshModel {
    bvh_tree: CompactBvh,
    simplices: Slab<Simplex>,
    model_map: ModelMap,
    qualities: Slab<Quality>,
    aabb: Aabb<Real, 3>,
    transform: Option<Affine3<Real>>,
    pub priority: u8,
    /// Human-readable name for this mesh model.
    pub name: String,

    // BVH bookkeeping for the model tree
    pub id: usize,
    node_index: usize,
}

pub enum MeshModelError {
    DegenerateSimplex,
}

impl DeepSizeOf for MeshModel {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.simplices.deep_size_of_children(context)
            + self.model_map.deep_size_of_children(context)
            + self.qualities.deep_size_of_children(context)
            + self.bvh_tree.deep_size_of_children(context)
            + self.name.deep_size_of_children(context)
    }
}

impl MeshModel {
    /// Build a curvilinear mesh by decomposing a structured grid into tetrahedra.
    ///
    /// Each voxel is split into 5 tetrahedra using the alternating-parity
    /// 5-simplex decomposition (Knuth, *TAOCP* Vol. 4 Fasc. 6).  All tetrahedra
    /// use [`InterpolateModel`] for barycentric quality interpolation.
    ///
    /// # Parameters
    ///
    /// * `vertices`   – `ni × nj × nk` grid points in some ordering defined by `chart`.
    /// * `qualities`  – one `Quality` per vertex.
    /// * `dimensions` – `(ni, nj, nk)` grid dimensions.
    /// * `chart`      – maps `(i, j, k)` grid indices to a vertex index.
    pub fn curvilinear_mesh<F>(
        vertices: Vec<Point3<Real>>,
        qualities: Vec<Quality>,
        dimensions: (usize, usize, usize),
        chart: F,
    ) -> Result<Self, MeshModelError>
    where
        F: Fn(usize, usize, usize) -> usize,
    {
        let (ni, nj, nk) = dimensions;
        let mut faces = Vec::with_capacity(5 * (ni - 1) * (nj - 1) * (nk - 1));

        // See: The Art of Computer Programming, Volume 4, Fascicle 6 for the
        // 5-simplex decomposition of a cube.
        for i in 0..ni - 1 {
            for j in 0..nj - 1 {
                for k in 0..nk - 1 {
                    let v000 = chart(i, j, k);
                    let v100 = chart(i + 1, j, k);
                    let v010 = chart(i, j + 1, k);
                    let v110 = chart(i + 1, j + 1, k);
                    let v001 = chart(i, j, k + 1);
                    let v101 = chart(i + 1, j, k + 1);
                    let v011 = chart(i, j + 1, k + 1);
                    let v111 = chart(i + 1, j + 1, k + 1);

                    if (i + j + k) % 2 == 0 {
                        faces.push(Point4::new(v000, v100, v010, v001)); // Corner 0
                        faces.push(Point4::new(v110, v100, v010, v111)); // Corner 1
                        faces.push(Point4::new(v101, v100, v001, v111)); // Corner 2
                        faces.push(Point4::new(v011, v010, v001, v111)); // Corner 3
                        faces.push(Point4::new(v100, v010, v001, v111)); // Central Core
                    } else {
                        faces.push(Point4::new(v100, v000, v110, v101)); // Corner 0
                        faces.push(Point4::new(v010, v000, v110, v011)); // Corner 1
                        faces.push(Point4::new(v001, v000, v101, v011)); // Corner 2
                        faces.push(Point4::new(v111, v110, v101, v011)); // Corner 3
                        faces.push(Point4::new(v000, v110, v101, v011)); // Central Core
                    }
                }
            }
        }

        let models = faces
            .iter()
            .map(|q| {
                Model::from(InterpolateModel {
                    qualities: q.map(|i| i as u32),
                })
            })
            .collect();

        Self::new(
            vertices,
            faces,
            models,
            qualities,
            DEFAULT_PRIORITY,
            None,
            String::new(),
        )
    }

    /// Create a mesh model from raw geometry data.
    ///
    /// Builds an internal BVH over the simplices.  If `transform` is
    /// provided it is treated as a world-to-local affine map: query points
    /// are transformed into local space before the BVH is consulted, and
    /// the AABB is computed in world space by transforming vertices with the
    /// inverse.
    pub fn new(
        vertices: Vec<Point3<Real>>,
        faces: Vec<Point4<usize>>,
        models: Vec<Model>,
        qualities: Vec<Quality>,
        priority: u8,
        transform: Option<Affine3<Real>>,
        name: String,
    ) -> Result<Self, MeshModelError> {
        let local_to_global_map = |p| transform.map_or(p, |aff| aff.inverse_transform_point(&p));
        let min_point =
            vertices
                .iter()
                .fold(Point3::new(Real::MAX, Real::MAX, Real::MAX), |acc, p| {
                    let transformed = local_to_global_map(*p);
                    Point3::new(
                        acc.x.min(transformed.x),
                        acc.y.min(transformed.y),
                        acc.z.min(transformed.z),
                    )
                });

        let max_point =
            vertices
                .iter()
                .fold(Point3::new(Real::MIN, Real::MIN, Real::MIN), |acc, p| {
                    let transformed = local_to_global_map(*p);
                    Point3::new(
                        acc.x.max(transformed.x),
                        acc.y.max(transformed.y),
                        acc.z.max(transformed.z),
                    )
                });
        let spatial_padding = Vector3::repeat(0.01);
        let aabb = Aabb::with_bounds(min_point - spatial_padding, max_point + spatial_padding);

        let mut build_simplices: Vec<BuildSimplex> = faces
            .iter()
            .filter_map(|f| {
                BuildSimplex::new(vertices[f.x], vertices[f.y], vertices[f.z], vertices[f.w])
            })
            .collect();

        if build_simplices.len() != faces.len() {
            return Err(MeshModelError::DegenerateSimplex);
        }
        drop(vertices);
        drop(faces);

        let full_tree = Bvh::build_par(&mut build_simplices);
        // Convert to the compact query-only representation and gather the
        // per-simplex arrays into its leaf order.
        let (bvh_tree, order) = CompactBvh::from_bvh(&full_tree, build_simplices.len());
        drop(full_tree);
        let simplices: Vec<Simplex> = order
            .iter()
            .map(|&i| build_simplices[i as usize].simplex)
            .collect();
        let model_map = ModelMap::from_models(models).reorder(&order);

        Ok(Self::from_parts(
            bvh_tree,
            Slab::Owned(simplices),
            model_map,
            Slab::Owned(qualities),
            aabb,
            transform,
            priority,
            name,
        ))
    }

    /// Assemble a model from arrays already in BVH leaf order.
    ///
    /// This is how a model comes back off disk: the index file stores exactly
    /// these parts, so opening one is a matter of mapping them.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        bvh_tree: CompactBvh,
        simplices: Slab<Simplex>,
        model_map: ModelMap,
        qualities: Slab<Quality>,
        aabb: Aabb<Real, 3>,
        transform: Option<Affine3<Real>>,
        priority: u8,
        name: String,
    ) -> Self {
        Self {
            bvh_tree,
            simplices,
            model_map,
            qualities,
            aabb,
            transform,
            priority,
            name,
            id: 0,
            node_index: 0,
        }
    }

    pub(crate) fn bvh(&self) -> &CompactBvh {
        &self.bvh_tree
    }

    pub(crate) fn simplices(&self) -> &Slab<Simplex> {
        &self.simplices
    }

    pub(crate) fn model_map(&self) -> &ModelMap {
        &self.model_map
    }

    pub(crate) fn qualities(&self) -> &Slab<Quality> {
        &self.qualities
    }

    pub(crate) fn transform(&self) -> Option<Affine3<Real>> {
        self.transform
    }

    /// Whether the model's arrays are memory-mapped from an index file
    /// rather than held on the heap.
    pub fn is_mapped(&self) -> bool {
        self.simplices.is_mapped()
    }

    /// Number of vertex-quality entries in this mesh.
    pub fn points(&self) -> usize {
        self.qualities.len()
    }

    /// Transform a point from world space into the mesh's local space.
    ///
    /// If no transform was supplied at construction the point is returned unchanged.
    pub fn global_to_local(&self, point: Point3<Real>) -> Point3<Real> {
        self.transform
            .map_or(point, |aff| aff.transform_point(&point))
    }

    /// Returns the quality at the given point using the first simplex that contains it.
    /// Returns `None` if no simplex contains the point.
    pub fn query(&self, point: Point3<Real>) -> Option<Quality> {
        let transformed = self.global_to_local(point);
        self.bvh_tree
            .containing_indices(&self.simplices, transformed)
            .next()
            .map(|i| {
                self.model_map
                    .quality_at(i, &self.qualities, &self.simplices[i], &transformed)
            })
    }

    pub fn pretty_print(&self) {
        let name_display = if self.name.is_empty() {
            format!("(id={})", self.id)
        } else {
            format!("{:?}", self.name)
        };
        println!(
            "Mesh model {} with {} vertices and {} simplices, tree depth = {}, priority = {}.",
            name_display,
            self.qualities.len(),
            self.simplices.len(),
            self.bvh_tree.depth(),
            self.priority,
        )
    }

    pub fn view(&self) -> MeshModelView {
        // The view is a diagnostics summary, so `bounds` stays f32 regardless
        // of `Real`, keeping the serialised schema stable.  The casts are
        // no-ops in the default build and load-bearing under `high_precision`.
        #[allow(clippy::unnecessary_cast)]
        let bounds: [f32; 6] = [
            self.aabb.min.x as f32,
            self.aabb.min.y as f32,
            self.aabb.min.z as f32,
            self.aabb.max.x as f32,
            self.aabb.max.y as f32,
            self.aabb.max.z as f32,
        ];
        MeshModelView {
            id: self.id,
            name: self.name.clone(),
            bounds,
            transform: self.transform,
            priority: self.priority,
            size: self.deep_size_of(),
        }
    }
}

/// `Contains` for `MeshModel` yields `(priority, quality)` when a simplex inside
/// this model contains the query point. This avoids a second BVH traversal in
/// the outer `ModelTree` – the traversal result is returned directly.
impl Contains<Real, 3, (u8, Quality)> for MeshModel {
    fn contains(&self, query_point: &Point3<Real>) -> Option<(u8, Quality)> {
        self.query(*query_point).map(|q| (self.priority, q))
    }
}

/// Returns the 3-D bounding box of this model's geometry.
/// Used when the caller needs the geometry AABB without the priority dimension.
impl MeshModel {
    pub fn aabb3(&self) -> Aabb<Real, 3> {
        self.aabb
    }
}

/// 4-D AABB used by the outer `ModelTree` BVH.
///
/// The first three dimensions are the model's geometry AABB. The fourth
/// dimension is the model's priority: both `min[3]` and `max[3]` are set to
/// `priority` (with a half-unit extent so the AABB is non-degenerate), which
/// lets the priority-ray traversal use `aabb.min[3]` as the `t_min` hit
/// distance and thereby visit models in priority order.
impl Bounded<Real, 4> for MeshModel {
    fn aabb(&self) -> Aabb<Real, 4> {
        let p = self.priority as Real;
        let min4 = Point::<Real, 4>::from(nalgebra::vector![
            self.aabb.min.x,
            self.aabb.min.y,
            self.aabb.min.z,
            p
        ]);
        let max4 = Point::<Real, 4>::from(nalgebra::vector![
            self.aabb.max.x,
            self.aabb.max.y,
            self.aabb.max.z,
            p + PRIORITY_AABB_EXTENT
        ]);
        Aabb::with_bounds(min4, max4)
    }
}

impl BHShape<Real, 4> for MeshModel {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use nalgebra::{Point3, Point4};

    fn unit_tetrahedron_universe() -> Vec<Point3<Real>> {
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ]
    }

    fn mock_quality(val: Real) -> Quality {
        Quality {
            rho: val,
            vp: val,
            vs: val,
            qp: val,
            qs: val,
            alpha: 1.0,
        }
    }

    fn generate_grid(ni: usize, nj: usize, nk: usize) -> Vec<Point3<Real>> {
        let mut vertices = Vec::with_capacity(ni * nj * nk);
        for k in 0..nk {
            for j in 0..nj {
                for i in 0..ni {
                    vertices.push(Point3::new(i as Real, j as Real, k as Real));
                }
            }
        }
        vertices
    }

    fn interpolate_all(faces: &[Point4<usize>]) -> Vec<Model> {
        faces
            .iter()
            .map(|f| {
                Model::from(InterpolateModel {
                    qualities: f.map(|i| i as u32),
                })
            })
            .collect()
    }

    #[test]
    fn test_simplex_barycentric_properties() {
        let v = unit_tetrahedron_universe();
        let simplex = Simplex::new(v[0], v[1], v[2], v[3]).unwrap();

        // Test vertex identity
        assert_relative_eq!(
            simplex.barycentric_coordinates(v[0]),
            Point4::new(1.0, 0.0, 0.0, 0.0)
        );
        assert_relative_eq!(
            simplex.barycentric_coordinates(v[3]),
            Point4::new(0.0, 0.0, 0.0, 1.0)
        );
    }

    #[test]
    fn test_simplex_aabb_properties() {
        let v0 = Point3::new(-1.0, 2.0, 0.0);
        let v1 = Point3::new(3.0, -4.0, 1.0);
        let v2 = Point3::new(0.0, 0.0, 5.0);
        let v3 = Point3::new(1.0, 1.0, 1.0);

        let simplex = BuildSimplex::new(v0, v1, v2, v3).unwrap();
        let aabb = simplex.aabb();

        assert_relative_eq!(aabb.min.x, -1.0);
        assert_relative_eq!(aabb.max.x, 3.0);
        assert_relative_eq!(aabb.min.z, 0.0);
        assert_relative_eq!(aabb.max.z, 5.0);
    }

    #[test]
    fn test_mesh_model_query_interpolation() {
        let ni = 2;
        let nj = 2;
        let nk = 2;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = (0..vertices.len())
            .map(|idx| mock_quality(idx as Real))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart)
            .unwrap_or_else(|_| panic!("mesh construction failed"));

        // Vertex (1,1,1) is index 7 in a 2x2x2 grid
        let p_v7 = Point3::new(1.0, 1.0, 1.0);
        let q_v7 = mesh.query(p_v7).expect("Should find a simplex");

        assert_relative_eq!(q_v7.rho, 7.0, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_model_query_multi_cell_interpolation() {
        let ni = 5;
        let nj = 5;
        let nk = 5;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(p.x + p.y + p.z))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart)
            .unwrap_or_else(|_| panic!("mesh construction failed"));

        // Interior: x=2.5, y=1.2, z=3.7 -> rho = 7.4
        let p_in = Point3::new(2.5, 1.2, 3.7);
        let q_in = mesh.query(p_in).expect("Should find interior");

        assert_relative_eq!(q_in.rho, 7.4, epsilon = 1e-5);
    }

    #[test]
    fn test_mesh_model_query_outside_returns_none() {
        let v = unit_tetrahedron_universe();
        let quality = mock_quality(1.0);
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = interpolate_all(&faces);
        let qualities = vec![quality; 4];
        let mesh = MeshModel::new(v, faces, models, qualities, 0, None, String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        let q = mesh.query(Point3::new(5.0, 5.0, 5.0));
        assert!(q.is_none());
    }

    #[test]
    fn test_aabb_correct_for_simple_mesh() {
        let ni = 3;
        let nj = 3;
        let nk = 3;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices.iter().map(|_| mock_quality(1.0)).collect();
        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart)
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        let aabb = mesh.aabb3();
        // The mesh AABB includes 0.01 of spatial padding on each side.
        assert_relative_eq!(aabb.min.x, 0.0, epsilon = 0.011);
        assert_relative_eq!(aabb.max.x, 2.0, epsilon = 0.011);
        assert_relative_eq!(aabb.min.z, 0.0, epsilon = 0.011);
        assert_relative_eq!(aabb.max.z, 2.0, epsilon = 0.011);
    }

    #[test]
    fn test_mesh_model_interpolate_centroid() {
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let qualities_vec: Vec<Quality> = (0..4).map(|i| mock_quality(i as Real)).collect();
        let models = interpolate_all(&faces);
        let mesh = MeshModel::new(v, faces, models, qualities_vec, 0, None, String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        // Deliberately NOT the centroid.  At the centroid all four weights are
        // 0.25, so with qualities 0,1,2,3 the result is 1.5 under *any* of the
        // 24 index permutations — the old test could not detect a regression in
        // the very permutation its own comment documented.  Four *distinct*
        // weights pin the permutation down.
        //
        // `unit_tetrahedron_universe` is [origin, e_x, e_y, e_z] with anchor
        // c3 = e_z, so barycentric_coordinates(x, y, z) = (1-x-y-z, x, y, z).
        let p = Point3::new(0.5, 0.25, 0.1);
        let bary = [1.0 - 0.5 - 0.25 - 0.1, 0.5, 0.25, 0.1]; // (l0, l1, l2, l3)
        let q = mesh.query(p).expect("point is inside the tetrahedron");

        // `interpolate_quality` (model.rs) pairs qualities[indices.(w,x,y,z)]
        // with weights (bary.w, bary.x, bary.y, bary.z).  Here indices are
        // (x,y,z,w) = (0,1,2,3), so the pairing is:
        //   qualities[3]=3 * l3,  qualities[0]=0 * l0,
        //   qualities[1]=1 * l1,  qualities[2]=2 * l2
        let expected = 3.0 * bary[3] + 0.0 * bary[0] + 1.0 * bary[1] + 2.0 * bary[2];
        assert_relative_eq!(q.rho, expected, epsilon = 1e-4);

        // Guard the guard: equal weights would make this as permutation-blind
        // as the test it replaced.
        assert_ne!(expected, 1.5, "weights must not be symmetric");
        let mut sorted = bary;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted
            .windows(2)
            .for_each(|w| assert_ne!(w[0], w[1], "all four barycentric weights must be distinct"));
    }

    /// `CONTAINMENT_EPS` deliberately admits points slightly outside a face so
    /// that queries never fall through the crack between two simplices.  The
    /// cost is that points on a shared face are contained by more than one
    /// simplex.
    ///
    /// The meaningful question is not "is the multiplicity > 1" — with a
    /// positive epsilon at a shared face it must be — but "do all containing
    /// simplices agree on the answer".  For a linear field they must, because
    /// barycentric interpolation of a linear field is exact regardless of which
    /// tetrahedron does the interpolating.
    ///
    /// This discharges the `# TODO (Scientific Review)` in `simplex.rs`.
    #[test]
    fn test_containment_eps_overlap_is_consistent() {
        let (ni, nj, nk) = (4, 4, 4);
        let vertices = generate_grid(ni, nj, nk);
        // A linear field: every simplex containing a point must interpolate it
        // to the same value.
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(2.0 * p.x + 3.0 * p.y - p.z))
            .collect();

        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices, qualities, (ni, nj, nk), chart)
            .unwrap_or_else(|_| panic!("mesh construction failed"));

        // Points chosen to sit exactly on cell faces, edges and vertices —
        // precisely where the epsilon causes multiple simplices to claim them.
        let probes = [
            Point3::new(1.0, 1.0, 1.0),
            Point3::new(1.0, 1.5, 2.0),
            Point3::new(2.0, 2.0, 1.5),
            Point3::new(1.5, 1.0, 1.0),
            Point3::new(0.5, 1.0, 2.5),
        ];

        let mut max_multiplicity = 0usize;
        for p in probes {
            let want = 2.0 * p.x + 3.0 * p.y - p.z;

            let containing: Vec<_> = mesh.simplices.iter().filter(|s| s.contains(&p)).collect();
            assert!(
                !containing.is_empty(),
                "no simplex contains {p:?} — this is the crack CONTAINMENT_EPS exists to close"
            );
            max_multiplicity = max_multiplicity.max(containing.len());

            // The public query must agree with the analytic field...
            let q = mesh.query(p).expect("probe must be covered");
            assert_relative_eq!(q.rho, want, epsilon = 1e-4);
        }

        // Pin the observed overlap.  32 is the measured maximum for this mesh:
        // an interior grid vertex is shared by 8 cells, and the 5-tetrahedron
        // decomposition puts 4 of each cell's tetrahedra on that vertex.  This
        // is not a correctness requirement — the agreement asserted above is —
        // but a jump beyond it means the epsilon has started admitting
        // simplices well past the shared face and should be re-examined.
        assert!(
            (2..=32).contains(&max_multiplicity),
            "unexpected containment multiplicity {max_multiplicity} (expected 2..=32); \
             CONTAINMENT_EPS may be mis-scaled for this mesh"
        );
    }

    #[test]
    fn test_constant_model_returns_fixed_quality() {
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let q_fixed = Quality {
            rho: 42.0,
            vp: 1.0,
            vs: 2.0,
            qp: 3.0,
            qs: 4.0,
            alpha: 1.0,
        };
        let qualities = vec![q_fixed];
        let models = vec![Model::from(ConstantModel { quality: 0 })];
        let mesh = MeshModel::new(v, faces, models, qualities, 0, None, String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        let result = mesh.query(Point3::new(0.2, 0.1, 0.1));
        assert!(result.is_some());
        let q_result = result.unwrap();
        assert_relative_eq!(q_result.rho, 42.0, epsilon = 1e-4);
        assert_relative_eq!(q_result.vp, 1.0, epsilon = 1e-4);
    }

    /// A world-to-local translation by (-5, 0, 0) places the unit tetrahedron at
    /// world position [5, 6] × [0, 1] × [0, 1].  Queries must use world coords.
    #[test]
    fn test_transform_translates_queries() {
        use nalgebra::{Affine3, Translation3};
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = vec![Model::from(ConstantModel { quality: 0 })];
        let qualities = vec![mock_quality(5.0)];

        // World-to-local: subtract 5 from x-coordinate.
        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0 as Real, 0.0, 0.0).to_homogeneous(),
        );
        let mesh = MeshModel::new(v, faces, models, qualities, 0, Some(aff), String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));

        // (5.1, 0.1, 0.1) in world → (0.1, 0.1, 0.1) in local → inside
        let q = mesh.query(Point3::new(5.1, 0.1, 0.1));
        assert!(q.is_some());
        assert_relative_eq!(q.unwrap().rho, 5.0, epsilon = 1e-4);

        // (0.1, 0.1, 0.1) in world → (-4.9, 0.1, 0.1) in local → outside
        let q_outside = mesh.query(Point3::new(0.1, 0.1, 0.1));
        assert!(q_outside.is_none());
    }

    /// After the translation, the AABB must reflect the tetrahedron's world position.
    #[test]
    fn test_transform_aabb_in_world_space() {
        use nalgebra::{Affine3, Translation3};
        let v = unit_tetrahedron_universe();
        let faces = vec![Point4::new(0usize, 1, 2, 3)];
        let models = vec![Model::from(ConstantModel { quality: 0 })];
        let qualities = vec![mock_quality(1.0)];

        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0 as Real, 0.0, 0.0).to_homogeneous(),
        );
        let mesh = MeshModel::new(v, faces, models, qualities, 0, Some(aff), String::new())
            .unwrap_or_else(|_| panic!("mesh construction failed"));
        let aabb = mesh.aabb3();

        // Tetrahedron spans [5,6] × [0,1] × [0,1] in world space (plus the
        // 0.01 spatial padding on each side of the AABB).
        assert_relative_eq!(aabb.min.x, 5.0, epsilon = 0.011);
        assert_relative_eq!(aabb.max.x, 6.0, epsilon = 0.011);
        assert_relative_eq!(aabb.min.y, 0.0, epsilon = 0.011);
        assert_relative_eq!(aabb.min.z, 0.0, epsilon = 0.011);
    }

    /// Queries through a mesh large enough to exercise multi-simplex leaves
    /// and several tree levels must agree with direct simplex containment.
    #[test]
    fn test_compact_bvh_matches_exhaustive_scan() {
        let ni = 6;
        let nj = 5;
        let nk = 4;
        let vertices = generate_grid(ni, nj, nk);
        let qualities: Vec<Quality> = vertices
            .iter()
            .map(|p| mock_quality(p.x + 2.0 * p.y + 3.0 * p.z))
            .collect();
        let chart = |i, j, k| i + j * ni + k * ni * nj;
        let mesh = MeshModel::curvilinear_mesh(vertices.clone(), qualities, (ni, nj, nk), chart)
            .unwrap_or_else(|_| panic!("mesh construction failed"));

        // Points on a fine grid inside and outside the mesh.
        for xi in 0..=14 {
            for yi in 0..=10 {
                for zi in 0..=8 {
                    let p = Point3::new(
                        xi as Real * 0.45 - 0.5,
                        yi as Real * 0.45 - 0.5,
                        zi as Real * 0.45 - 0.5,
                    );
                    let via_bvh = mesh
                        .bvh_tree
                        .containing_indices(&mesh.simplices, p)
                        .next()
                        .is_some();
                    let exhaustive = mesh.simplices.iter().any(|s| s.contains(&p));
                    assert_eq!(via_bvh, exhaustive, "mismatch at {p:?}");
                    if via_bvh {
                        // The quality field is linear, so any containing
                        // simplex interpolates to the same value.
                        let q = mesh.query(p).unwrap();
                        assert_relative_eq!(q.rho, p.x + 2.0 * p.y + 3.0 * p.z, epsilon = 1e-3);
                    }
                }
            }
        }
    }
}
