use crate::compact_bvh::CompactBvh;
use crate::model::{AnyField, Field, Interpolated};
use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::{BuildSimplex, Simplex};
use deepsize::DeepSizeOf;

use bvh::aabb::Aabb;
use bvh::bounding_hierarchy::BoundingHierarchy;
use bvh::bvh::Bvh;
use nalgebra::{Affine3, Point3, Point4, Vector3};

/// The geometry of a tetrahedral mesh, indexed for point location.
///
/// The simplices are stored in BVH leaf order (the tree is built with the
/// `bvh` crate and then converted to a [`CompactBvh`]), so a leaf visit
/// scans adjacent entries of `simplices`.
pub struct TetMesh {
    bvh: CompactBvh,
    simplices: Vec<Simplex>,
    /// Vertex indices of each simplex, in the same order as `simplices`.
    faces: Vec<[u32; 4]>,
    num_vertices: usize,
    /// World-space bounds.
    aabb: Aabb<Real, 3>,
    /// World-to-local map applied to query points.
    transform: Option<Affine3<Real>>,
}

impl TetMesh {
    /// Index the simplices `faces` (four indices into `vertices` each).
    ///
    /// If `transform` is given it is a world-to-local affine map: query
    /// points are transformed into local space before the BVH is consulted,
    /// and the AABB is computed in world space by transforming vertices with
    /// the inverse.
    ///
    /// Also returns the BVH leaf order: simplex `i` of the mesh is
    /// `faces[order[i]]`.
    fn new(
        vertices: Vec<Point3<Real>>,
        faces: Vec<[u32; 4]>,
        transform: Option<Affine3<Real>>,
    ) -> Result<(Self, Vec<u32>), MeshModelError> {
        let to_world =
            |p: &Point3<Real>| transform.map_or(*p, |aff| aff.inverse_transform_point(p));
        let (min, max) = vertices.iter().map(to_world).fold(
            (Point3::from([Real::MAX; 3]), Point3::from([Real::MIN; 3])),
            |(min, max), p| (min.inf(&p), max.sup(&p)),
        );
        let padding = Vector3::repeat(0.01);
        let aabb = Aabb::with_bounds(min - padding, max + padding);

        let corner = |v: u32| vertices.get(v as usize).copied();
        let mut build_simplices = faces
            .iter()
            .map(|&[a, b, c, d]| {
                let [a, b, c, d] = [corner(a)?, corner(b)?, corner(c)?, corner(d)?];
                BuildSimplex::new(a, b, c, d)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(MeshModelError::DegenerateSimplex)?;
        // Each intermediate is dropped as soon as it is consumed, not at the
        // end of the function: the vertices before the BVH is built, the full
        // BVH before the simplices are gathered, and the build simplices
        // before the faces are.  On large meshes this lowers peak memory.
        let num_vertices = vertices.len();
        drop(vertices);

        let full_tree = Bvh::build_par(&mut build_simplices);
        // Convert to the compact query-only representation and gather the
        // simplices into its leaf order.
        let (bvh, order) = CompactBvh::from_bvh(&full_tree, build_simplices.len());
        drop(full_tree);
        let simplices = order
            .iter()
            .map(|&i| build_simplices[i as usize].simplex)
            .collect();
        drop(build_simplices);
        let faces = order.iter().map(|&i| faces[i as usize]).collect();
        let mesh = Self {
            bvh,
            simplices,
            faces,
            num_vertices,
            aabb,
            transform,
        };
        Ok((mesh, order))
    }

    /// The first simplex containing `point` (in world space), and the
    /// point's barycentric coordinates in it.
    pub fn locate(&self, point: Point3<Real>) -> Option<(usize, Point4<Real>)> {
        let local = self
            .transform
            .map_or(point, |aff| aff.transform_point(&point));
        self.bvh.containing_simplices(&self.simplices, local).next()
    }

    /// The vertex indices of simplex `simplex`.
    pub fn face(&self, simplex: usize) -> [u32; 4] {
        self.faces[simplex]
    }

    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    pub fn num_simplices(&self) -> usize {
        self.simplices.len()
    }

    /// World-space bounding box.
    pub fn aabb(&self) -> Aabb<Real, 3> {
        self.aabb
    }

    pub fn transform(&self) -> Option<Affine3<Real>> {
        self.transform
    }

    /// Depth of the BVH.
    pub fn depth(&self) -> usize {
        self.bvh.depth()
    }

    /// Bytes held on the heap.
    pub fn heap_size(&self) -> usize {
        self.simplices.capacity() * std::mem::size_of::<Simplex>()
            + self.faces.capacity() * std::mem::size_of::<[u32; 4]>()
            + self.bvh.deep_size_of()
            - std::mem::size_of::<CompactBvh>()
    }
}

/// Reasons a [`MeshModel`] cannot be built.
#[derive(Debug, PartialEq)]
pub enum MeshModelError {
    /// A simplex is flat, or refers to a vertex that doesn't exist.
    DegenerateSimplex,
    /// The field lacks data for some vertex or simplex of the mesh.
    FieldCoverage,
}

impl std::fmt::Display for MeshModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::DegenerateSimplex => "degenerate or out-of-range simplex",
            Self::FieldCoverage => "the field doesn't cover every vertex or simplex of the mesh",
        })
    }
}

/// A quality field over a tetrahedral mesh.
///
/// `F` is the field type: concrete where it is known at compile time (e.g.
/// [`MeshModel::curvilinear_mesh`]), and [`AnyField`] where it is chosen at
/// run time, as for the meshes a
/// [`ModelTree`](crate::model_tree::ModelTree) holds.
pub struct MeshModel<F = AnyField> {
    mesh: TetMesh,
    field: F,
}

impl<F: Field> MeshModel<F> {
    /// Pair the simplices `faces` (four indices into `vertices` each) with a
    /// field over them.  Per-simplex field data follows the order of
    /// `faces`.
    ///
    /// See [`TetMesh`] for `transform`.
    pub fn new(
        vertices: Vec<Point3<Real>>,
        faces: Vec<[u32; 4]>,
        mut field: F,
        transform: Option<Affine3<Real>>,
    ) -> Result<Self, MeshModelError> {
        if !field.covers(vertices.len(), faces.len()) {
            return Err(MeshModelError::FieldCoverage);
        }
        let (mesh, order) = TetMesh::new(vertices, faces, transform)?;
        field.reorder(&order);
        Ok(Self { mesh, field })
    }

    /// The quality at `point`, from the first simplex that contains it, or
    /// `None` if none does.
    pub fn query(&self, point: Point3<Real>) -> Option<Quality> {
        self.mesh.locate(point).map(|(simplex, bary)| {
            self.field
                .quality_at(simplex, self.mesh.face(simplex), &bary)
        })
    }

    pub fn mesh(&self) -> &TetMesh {
        &self.mesh
    }

    /// World-space bounding box.
    pub fn aabb3(&self) -> Aabb<Real, 3> {
        self.mesh.aabb()
    }

    /// Bytes held on the heap.
    pub fn heap_size(&self) -> usize {
        self.mesh.heap_size() + self.field.heap_size()
    }

    /// Forget the field's concrete type.
    pub fn into_any(self) -> MeshModel<AnyField>
    where
        F: Into<AnyField>,
    {
        MeshModel {
            mesh: self.mesh,
            field: self.field.into(),
        }
    }
}

impl MeshModel<Interpolated> {
    /// Build a curvilinear mesh by decomposing a structured grid into tetrahedra.
    ///
    /// Each voxel is split into 5 tetrahedra using the alternating-parity
    /// 5-simplex decomposition (Knuth, *TAOCP* Vol. 4 Fasc. 6), and the
    /// field interpolates the vertex qualities.
    ///
    /// # Parameters
    ///
    /// * `vertices`   – `ni × nj × nk` grid points in some ordering defined by `chart`.
    /// * `qualities`  – one `Quality` per vertex.
    /// * `dimensions` – `(ni, nj, nk)` grid dimensions.
    /// * `chart`      – maps `(i, j, k)` grid indices to a vertex index.
    pub fn curvilinear_mesh<C>(
        vertices: Vec<Point3<Real>>,
        qualities: Vec<Quality>,
        dimensions: (usize, usize, usize),
        chart: C,
    ) -> Result<Self, MeshModelError>
    where
        C: Fn(usize, usize, usize) -> usize,
    {
        let (ni, nj, nk) = dimensions;
        let mut faces: Vec<[u32; 4]> = Vec::with_capacity(5 * (ni - 1) * (nj - 1) * (nk - 1));
        let chart = |i, j, k| chart(i, j, k) as u32;

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
                        faces.push([v000, v100, v010, v001]); // Corner 0
                        faces.push([v110, v100, v010, v111]); // Corner 1
                        faces.push([v101, v100, v001, v111]); // Corner 2
                        faces.push([v011, v010, v001, v111]); // Corner 3
                        faces.push([v100, v010, v001, v111]); // Central Core
                    } else {
                        faces.push([v100, v000, v110, v101]); // Corner 0
                        faces.push([v010, v000, v110, v011]); // Corner 1
                        faces.push([v001, v000, v101, v011]); // Corner 2
                        faces.push([v111, v110, v101, v011]); // Corner 3
                        faces.push([v000, v110, v101, v011]); // Central Core
                    }
                }
            }
        }

        Self::new(vertices, faces, Interpolated::new(qualities), None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constant, DepthProfile, Profiled};
    use approx::assert_relative_eq;
    use bvh::aabb::Bounded;

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

    /// A one-tetrahedron mesh over the unit tetrahedron with `field`.
    fn unit_mesh<F: Field>(field: F, transform: Option<Affine3<Real>>) -> MeshModel<F> {
        MeshModel::new(
            unit_tetrahedron_universe(),
            vec![[0, 1, 2, 3]],
            field,
            transform,
        )
        .unwrap_or_else(|e| panic!("mesh construction failed: {e}"))
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
        let field = Interpolated::new(vec![mock_quality(1.0); 4]);
        let mesh = unit_mesh(field, None);
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
        let qualities: Vec<Quality> = (0..4).map(|i| mock_quality(i as Real)).collect();
        let mesh = unit_mesh(Interpolated::new(qualities), None);
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

        // `Interpolated` pairs the quality of the simplex's k-th vertex with
        // barycentric weight k.  Here the vertices are (0, 1, 2, 3), so the
        // pairing is qualities[k] = k with weight l_k.
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

            let containing: Vec<_> = mesh
                .mesh
                .simplices
                .iter()
                .filter(|s| s.contains(&p))
                .collect();
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
        let q_fixed = Quality {
            rho: 42.0,
            vp: 1.0,
            vs: 2.0,
            qp: 3.0,
            qs: 4.0,
            alpha: 1.0,
        };
        let mesh = unit_mesh(Constant::new(vec![q_fixed]), None);
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
        use nalgebra::Translation3;

        // World-to-local: subtract 5 from x-coordinate.
        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0 as Real, 0.0, 0.0).to_homogeneous(),
        );
        let mesh = unit_mesh(Constant::new(vec![mock_quality(5.0)]), Some(aff));

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
        use nalgebra::Translation3;

        let aff: Affine3<Real> = Affine3::from_matrix_unchecked(
            Translation3::new(-5.0 as Real, 0.0, 0.0).to_homogeneous(),
        );
        let mesh = unit_mesh(Constant::new(vec![mock_quality(1.0)]), Some(aff));
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
                        .mesh
                        .bvh
                        .containing_simplices(&mesh.mesh.simplices, p)
                        .next()
                        .is_some();
                    let exhaustive = mesh.mesh.simplices.iter().any(|s| s.contains(&p));
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

    #[test]
    fn test_profiled_mesh_queries_profile() {
        let profile =
            DepthProfile::new(&[0.0, 0.5], vec![mock_quality(10.0), mock_quality(20.0)]).unwrap();
        // Depth equals z; alpha 1 everywhere.
        let vertices = unit_tetrahedron_universe()
            .iter()
            .map(|p| [p.z, 1.0])
            .collect();
        let field = Profiled::new(vertices, profile);
        let mesh = unit_mesh(field, None);
        assert_relative_eq!(mesh.query(Point3::new(0.1, 0.1, 0.2)).unwrap().vs, 10.0);
        assert_relative_eq!(mesh.query(Point3::new(0.1, 0.1, 0.7)).unwrap().vs, 20.0);
        // Alpha comes from the vertices, not the profile layer.
        assert_relative_eq!(mesh.query(Point3::new(0.1, 0.1, 0.7)).unwrap().alpha, 1.0);
    }

    #[test]
    fn test_field_must_cover_the_mesh() {
        let unit = || (unit_tetrahedron_universe(), vec![[0, 1, 2, 3]]);
        let (v, faces) = unit();
        let constant = Constant::new(vec![mock_quality(1.0); 2]);
        let result = MeshModel::new(v, faces, constant, None);
        assert_eq!(result.err(), Some(MeshModelError::FieldCoverage));
        let (v, faces) = unit();
        let interpolated = Interpolated::new(vec![mock_quality(1.0); 3]);
        let result = MeshModel::new(v, faces, interpolated, None);
        assert_eq!(result.err(), Some(MeshModelError::FieldCoverage));
    }

    #[test]
    fn test_out_of_range_vertex_is_rejected() {
        let field = Constant::new(vec![mock_quality(1.0)]);
        let result = MeshModel::new(unit_tetrahedron_universe(), vec![[0, 1, 2, 4]], field, None);
        assert_eq!(result.err(), Some(MeshModelError::DegenerateSimplex));
    }

    #[test]
    fn test_into_any_keeps_queries() {
        let mesh = unit_mesh(Constant::new(vec![mock_quality(3.0)]), None).into_any();
        assert_relative_eq!(mesh.query(Point3::new(0.1, 0.1, 0.1)).unwrap().vs, 3.0);
    }
}
