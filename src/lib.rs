pub mod blend;
pub mod coastline;
pub mod compact_bvh;
pub mod index;
pub mod mesh;
pub mod model;
pub mod model_tree;
pub mod quality;
pub mod query;
pub mod real;
pub mod simplex;
pub mod slab;
pub mod surface;
mod tree_query;
pub mod triangle;
use pyo3::prelude::*;

#[pymodule]
mod nzcvm {
    use crate::coastline::{Coastline, Segment};
    use crate::index::{self, FINGERPRINT_LEN, IndexError};
    use crate::mesh::{MeshModel, MeshModelError};
    use crate::model::{ConstantModel, InterpolateModel, Model};
    use crate::model_tree::ModelTree;
    use crate::quality::Quality;
    use crate::query::Query;
    use crate::real::Real;
    use crate::surface::SurfaceModel;
    use std::path::PathBuf;

    use nalgebra::{Affine3, Matrix4, Point2, Point3, Point4};
    use ndarray::{Array1, Array2, Axis, array, azip};
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyReadonlyArray3, PyReadwriteArray2, PyUntypedArrayMethods,
    };
    use pyo3::exceptions::{PyIOError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;
    use pythonize::pythonize;

    fn index_error(e: IndexError) -> PyErr {
        match e {
            IndexError::Io(e) => PyIOError::new_err(e.to_string()),
            other => PyValueError::new_err(other.to_string()),
        }
    }

    fn fingerprint_array(bytes: &[u8]) -> PyResult<[u8; FINGERPRINT_LEN]> {
        bytes.try_into().map_err(|_| {
            PyValueError::new_err(format!(
                "fingerprint must be {FINGERPRINT_LEN} bytes, got {}",
                bytes.len()
            ))
        })
    }

    /// Open a compiled index as a memory-mapped mesh model.
    ///
    /// The header is read and checked; the record sections are mapped and
    /// paged in as queries touch them.
    #[pyfunction]
    pub fn mesh_model_open(path: PathBuf) -> PyResult<PyMeshModel> {
        Ok(PyMeshModel {
            inner: Some(MeshModel::open_index(&path).map_err(index_error)?),
        })
    }

    /// The source fingerprint recorded in the index at `path`.
    ///
    /// Reads only the header page.
    #[pyfunction]
    pub fn index_fingerprint<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyBytes>> {
        let fingerprint = index::read_fingerprint(&path).map_err(index_error)?;
        Ok(PyBytes::new(py, &fingerprint))
    }

    /// Coordinate arrays and optional boolean mask for a vectorised query.
    ///
    /// Bundles the three `(N,)` coordinate arrays and an optional `(N,)` boolean
    /// mask into a single object so that [`PyModelTree::query_many`] stays under
    /// Clippy's argument-count limit.  The object holds no GIL lock — arrays are
    /// borrowed only during the actual query call.
    #[pyclass]
    pub struct QueryCoordinates {
        pub x: Py<PyArray1<Real>>,
        pub y: Py<PyArray1<Real>>,
        pub z: Py<PyArray1<Real>>,
        pub mask: Option<Py<PyArray1<bool>>>,
    }

    #[pymethods]
    impl QueryCoordinates {
        /// Create a new ``QueryCoordinates``.
        ///
        /// Parameters
        /// ----------
        /// x, y, z :
        ///     Three 1-D float32 coordinate arrays of length *N*.
        /// mask :
        ///     Optional 1-D boolean array of length *N*.  When supplied, only
        ///     points where ``mask[i]`` is ``True`` are queried; other rows in
        ///     ``out`` are left untouched.
        #[new]
        #[pyo3(signature = (x, y, z, mask=None))]
        pub fn new(
            x: Py<PyArray1<Real>>,
            y: Py<PyArray1<Real>>,
            z: Py<PyArray1<Real>>,
            mask: Option<Py<PyArray1<bool>>>,
        ) -> Self {
            QueryCoordinates { x, y, z, mask }
        }
    }

    /// Parameters controlling a vectorised query (priority range).
    ///
    /// Bundles `priority_lo` and `priority_hi` into a single object so that
    /// [`PyModelTree::query_many`] has a compact signature.
    #[pyclass(from_py_object)]
    #[derive(Debug, Clone, Copy)]
    pub struct QueryParams {
        pub priority_lo: u8,
        pub priority_hi: u8,
    }

    #[pymethods]
    impl QueryParams {
        /// Create a new ``QueryParams``.
        ///
        /// Parameters
        /// ----------
        /// priority_lo, priority_hi :
        ///     Inclusive priority bounds; only models whose priority falls in
        ///     ``[priority_lo, priority_hi]`` are queried.  Pass ``0, 255`` to
        ///     query all models.
        #[new]
        pub fn new(priority_lo: u8, priority_hi: u8) -> Self {
            QueryParams {
                priority_lo,
                priority_hi,
            }
        }
    }

    /// Python-facing single tetrahedral mesh model.
    ///
    /// Wraps a [`MeshModel`] and exposes query, AABB, name and priority.
    /// Consumed (moved into a [`ModelTree`]) when passed to [`model_tree`].
    #[pyclass]
    pub struct PyMeshModel {
        inner: Option<MeshModel>,
    }

    /// Python-facing velocity model (a compiled [`ModelTree`]).
    #[pyclass]
    pub struct PyModelTree {
        pub inner: ModelTree,
    }

    /// Create a [`PyMeshModel`] from raw NumPy arrays.
    ///
    /// # Arguments
    ///
    /// * `vertices_py`   – `(N, 3)` float array of vertex coordinates.
    /// * `faces_py`      – `(M, 4)` integer array of tetrahedral cell indices.
    /// * `types_py`      – `(M,)` u8 array: `0` = constant, `1` = interpolate.
    /// * `models_py`     – flat integer look-up array whose stride varies with
    ///   model type.  For each entry in `types_py`:
    ///   - type `0` (constant): consume **1** index — the quality index.
    ///   - type `1` (interpolate): consume **4** indices — the four vertex
    ///     quality indices stored in `(x, y, z, w)` order matching the
    ///     corresponding simplex vertices.
    ///   The total length of `models_py` must equal
    ///   `sum(1 if t==0 else 4 for t in types_py)`.
    /// * `qualities_py`  – `(Q, 6)` array of quality values indexed by `models_py`.
    /// * `priority`      – model priority (lower number = higher priority).
    /// * `transform_py`  – optional 4×4 world-to-local affine transform.
    /// * `name`          – optional human-readable name for the model.
    #[pyfunction]
    #[pyo3(signature = (vertices_py, faces_py, types_py, models_py, qualities_py, priority, transform_py, name=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn mesh_model(
        vertices_py: PyReadonlyArray2<Real>,
        faces_py: PyReadonlyArray2<usize>,
        types_py: PyReadonlyArray1<u8>,
        models_py: PyReadonlyArray1<usize>,
        qualities_py: PyReadonlyArray2<Real>,
        priority: u8,
        transform_py: Option<PyReadonlyArray2<Real>>,
        name: Option<String>,
    ) -> PyResult<PyMeshModel> {
        let vertices = vertices_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point3::new(a[0], a[1], a[2]))
            .collect();
        let faces = faces_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point4::new(a[0], a[1], a[2], a[3]))
            .collect();

        // Build Vec<Quality> from the (N, 6) numpy array.
        let qualities: Vec<Quality> = qualities_py
            .as_array()
            .axis_iter(Axis(0))
            .map(Quality::from)
            .collect();

        // Quality indices are stored as u32 inside the compiled model.
        if qualities.len() > u32::MAX as usize {
            return Err(PyValueError::new_err(format!(
                "Too many qualities for a single mesh model: {}",
                qualities.len()
            )));
        }

        let types = types_py.as_array();
        let mut models_vec = Vec::with_capacity(types.len());
        let model_idx = models_py.as_array();
        let mut idx = 0;
        for model_type in types.iter() {
            match model_type {
                0 => {
                    models_vec.push(Model::from(ConstantModel {
                        quality: model_idx[idx] as u32,
                    }));
                    idx += 1;
                }
                1 => {
                    models_vec.push(Model::from(InterpolateModel {
                        qualities: Point4::new(
                            model_idx[idx] as u32,
                            model_idx[idx + 1] as u32,
                            model_idx[idx + 2] as u32,
                            model_idx[idx + 3] as u32,
                        ),
                    }));
                    idx += 4;
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "Invalid model type {}",
                        model_type
                    )));
                }
            }
        }
        if idx != model_idx.len() {
            return Err(PyValueError::new_err(format!(
                "Invalid model types detected (did not read all models from model array using types given). idx = {}, models len = {}",
                idx,
                model_idx.len()
            )));
        }

        let transform = transform_py.map(|arr| {
            Affine3::from_matrix_unchecked(Matrix4::from_iterator(arr.as_array().iter().cloned()))
        });
        let mesh_model = MeshModel::new(
            vertices,
            faces,
            models_vec,
            qualities,
            priority,
            transform,
            name.clone().unwrap_or_default(),
        );
        match mesh_model {
            Ok(mesh_model) => Ok(PyMeshModel {
                inner: Some(mesh_model),
            }),
            Err(MeshModelError::DegenerateSimplex) => Err(PyValueError::new_err(format!(
                "Degenerate simplex encountered in model: {}",
                name.clone().unwrap_or_default()
            ))),
        }
    }

    /// Combine one or more [`PyMeshModel`]s into a queryable [`PyModelTree`].
    ///
    /// Each `PyMeshModel` is consumed (its inner data is moved out) so it
    /// cannot be reused after this call.
    ///
    /// # Errors
    ///
    /// Returns `ValueError` if a `PyMeshModel` has already been consumed.
    #[pyfunction]
    pub fn model_tree(py: Python<'_>, mesh_models: Vec<Py<PyMeshModel>>) -> PyResult<PyModelTree> {
        let mut models = Vec::with_capacity(mesh_models.len());
        for m_py in mesh_models {
            let mut m = m_py.borrow_mut(py);
            let model = m.inner.take().ok_or_else(|| {
                PyValueError::new_err(
                    "MeshModel has already been consumed by a previous model_tree() call",
                )
            })?;
            models.push(model);
        }
        Ok(PyModelTree {
            inner: ModelTree::new(models),
        })
    }

    impl PyMeshModel {
        /// The wrapped model, or the error every method raises once
        /// `model_tree()` has moved it out.
        fn model(&self) -> PyResult<&MeshModel> {
            self.inner
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("MeshModel has been consumed by model_tree()"))
        }
    }

    #[pymethods]
    impl PyMeshModel {
        /// Write this model as a compiled index at `path`.
        ///
        /// `fingerprint` is the caller's 32-byte summary of the source mesh,
        /// stored in the header so a stale index can be recognised later.
        pub fn write_index(&self, path: PathBuf, fingerprint: &[u8]) -> PyResult<()> {
            let inner = self.model()?;
            let fingerprint = fingerprint_array(fingerprint)?;
            inner.write_index(&fingerprint, &path).map_err(index_error)
        }

        /// Whether the model's arrays are memory-mapped from an index file.
        pub fn is_mapped(&self) -> PyResult<bool> {
            let inner = self.model()?;
            Ok(inner.is_mapped())
        }

        /// Query the mesh model at a single point.
        ///
        /// Returns a dict with keys `rho`, `vp`, `vs`, `qp`, `qs`, `alpha`,
        /// or `None` if the point lies outside this model's region.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        pub fn query<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Option<Bound<'py, PyAny>>> {
            let inner = self.model()?;
            let pt = Point3::new(x, y, z);
            inner
                .query(pt)
                .map(|q| pythonize(py, &q).map_err(|e| e.into()))
                .transpose()
        }

        /// Return the 3-D axis-aligned bounding box of this mesh model.
        ///
        /// Returns a pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[allow(clippy::type_complexity)]
        pub fn aabb<'py>(
            &self,
            py: Python<'py>,
        ) -> PyResult<(Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>)> {
            let inner = self.model()?;
            let b = inner.aabb3();
            let min = array![b.min.x, b.min.y, b.min.z];
            let max = array![b.max.x, b.max.y, b.max.z];
            Ok((min.into_pyarray(py), max.into_pyarray(py)))
        }

        /// Human-readable name assigned at construction time.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[getter]
        pub fn name(&self) -> PyResult<String> {
            let inner = self.model()?;
            Ok(inner.name.clone())
        }

        /// Model priority (lower number = higher priority in the tree).
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        #[getter]
        pub fn priority(&self) -> PyResult<u8> {
            let inner = self.model()?;
            Ok(inner.priority)
        }

        /// Return a serialisable Python dict describing this mesh model.
        ///
        /// # Errors
        ///
        /// Returns `ValueError` if the model has been consumed by `model_tree()`.
        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self.model()?;
            pythonize(py, &inner.view()).map_err(|e| e.into())
        }
    }

    #[pymethods]
    impl PyModelTree {
        /// Query the velocity model at a single point.
        ///
        /// Only models whose priority falls in `[priority_lo, priority_hi]` contribute.
        /// Use `0, 255` to query all models.
        /// Returns a dict `{rho, vp, vs, qp, qs, alpha}` or `None` if outside all models.
        pub fn query<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
            priority_lo: u8,
            priority_hi: u8,
        ) -> PyResult<Option<Bound<'py, PyAny>>> {
            let pt = Point3::new(x, y, z);
            self.inner
                .query(pt, None, priority_lo, priority_hi)
                .map(|q| pythonize(py, &q).map_err(|e| e.into()))
                .transpose()
        }

        /// Return the combined 3-D axis-aligned bounding box of all mesh models.
        ///
        /// Returns a pair ``(min_xyz, max_xyz)`` of shape-``(3,)`` float32 arrays.
        pub fn aabb<'py>(
            &self,
            py: Python<'py>,
        ) -> (Bound<'py, PyArray1<Real>>, Bound<'py, PyArray1<Real>>) {
            let b = self.inner.aabb();
            let min = array![b.min.x, b.min.y, b.min.z];
            let max = array![b.max.x, b.max.y, b.max.z];
            (min.into_pyarray(py), max.into_pyarray(py))
        }

        /// Query with BVH traversal statistics (AABB tests, simplex tests, etc.).
        ///
        /// Returns a dict with keys `aabb_tests`, `simplex_tests`, `hit_count`,
        /// `output` (quality dict or `None`), and `elapsed` (nanoseconds).
        pub fn query_stats<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Bound<'py, PyAny>> {
            let pt = Point3::new(x, y, z);
            pythonize(py, &self.inner.query_stats(pt)).map_err(|e| e.into())
        }

        /// Return a full diagnostic breakdown listing each model's contribution.
        ///
        /// Returns a dict with keys `contributions` (list of dicts), `output`
        /// (quality dict or `None`), and `termination` (int or `None`).
        pub fn explain<'py>(
            &self,
            py: Python<'py>,
            x: Real,
            y: Real,
            z: Real,
        ) -> PyResult<Bound<'py, PyAny>> {
            let pt = Point3::new(x, y, z);
            pythonize(py, &self.inner.query_explain(pt)).map_err(|e| e.into())
        }

        /// Return a serialisable Python dict describing the model tree structure.
        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let view = self.inner.view();
            pythonize(py, &view).map_err(|e| e.into())
        }

        /// Query the model for many points at once, writing results into a
        /// caller-supplied output buffer (ufunc style).
        ///
        /// Python is responsible for allocating and zeroing `out` before
        /// calling; this avoids the allocation inside Rust and lets the caller
        /// reuse memory across calls.
        ///
        /// # Arguments
        ///
        /// * `out`    – `(N, 6)` float32 array that receives the results
        ///   in-place.  Columns are ordered `[rho, vp, vs, qp, qs, alpha]`.
        ///   Rows for points outside all matching models are left unchanged.
        /// * `coords` – [`QueryCoordinates`] bundling the three `(N,)` float32
        ///   coordinate arrays and an optional boolean mask.
        /// * `params` – priority bounds; use `QueryParams(0, 255)` for all models.
        ///
        /// The GIL is released during the hot loop so other Python threads
        /// can run concurrently.
        #[pyo3(signature = (out, coords, params))]
        pub fn query_many(
            &self,
            py: Python<'_>,
            mut out: PyReadwriteArray2<Real>,
            coords: &QueryCoordinates,
            params: QueryParams,
        ) -> PyResult<()> {
            let lo = params.priority_lo;
            let hi = params.priority_hi;
            // Borrow all array views while the GIL is held.
            // The views hold no Python token so they are safe to use after detach.
            let x_ro = coords.x.bind(py).readonly();
            let y_ro = coords.y.bind(py).readonly();
            let z_ro = coords.z.bind(py).readonly();
            let xs = x_ro.as_array();
            let ys = y_ro.as_array();
            let zs = z_ro.as_array();
            let n = xs.len();
            if ys.len() != n || zs.len() != n {
                return Err(PyValueError::new_err(format!(
                    "x, y, z must have the same length; got x={}, y={}, z={}",
                    n,
                    ys.len(),
                    zs.len(),
                )));
            }
            if out.shape()[0] != n || out.shape()[1] != 6 {
                return Err(PyValueError::new_err(format!(
                    "out must have shape ({n}, 6); got {:?}",
                    out.shape(),
                )));
            }
            let mut buf = out.as_array_mut();
            match &coords.mask {
                None => {
                    py.detach(|| {
                        azip!((mut lane in buf.rows_mut(), &xi in &xs, &yi in &ys, &zi in &zs) {
                            let pt = Point3::new(xi, yi, zi);
                            if let Some(q) = self.inner.query(pt, None, lo, hi) {
                                lane[0] = q.rho;
                                lane[1] = q.vp;
                                lane[2] = q.vs;
                                lane[3] = q.qp;
                                lane[4] = q.qs;
                                lane[5] = q.alpha;
                            }
                        });
                    });
                }
                Some(mask_py) => {
                    let m_ro = mask_py.bind(py).readonly();
                    let ms = m_ro.as_array();
                    if ms.len() != n {
                        return Err(PyValueError::new_err(format!(
                            "where mask must have length {n}; got {}",
                            ms.len(),
                        )));
                    }
                    py.detach(|| {
                        azip!((mut lane in buf.rows_mut(), &xi in &xs, &yi in &ys, &zi in &zs, &mi in &ms) {
                            if mi {
                                let pt = Point3::new(xi, yi, zi);
                                if let Some(q) = self.inner.query(pt, None, lo, hi) {
                                    lane[0] = q.rho;
                                    lane[1] = q.vp;
                                    lane[2] = q.vs;
                                    lane[3] = q.qp;
                                    lane[4] = q.qs;
                                    lane[5] = q.alpha;
                                }
                            }
                        });
                    });
                }
            }
            Ok(())
        }

        /// Print a human-readable summary of the model tree to stdout.
        pub fn print_structure(&self) {
            self.inner.pretty_print();
        }
    }

    /// Python-facing surface model for 2D elevation/property interpolation.
    #[pyclass]
    pub struct PySurfaceModel {
        pub inner: Option<SurfaceModel>,
    }

    /// Create a [`PySurfaceModel`] from raw NumPy arrays.
    ///
    /// # Arguments
    /// * `vertices_py` – `(N, 2)` array of (x, y) coordinates.
    /// * `faces_py`    – `(M, 3)` array of triangle vertex indices.
    /// * `z_py`        – `(N,)` array of values (e.g. elevation) at each vertex.
    #[pyfunction]
    pub fn surface_model(
        vertices_py: PyReadonlyArray2<Real>,
        faces_py: PyReadonlyArray2<usize>,
        z_py: PyReadonlyArray1<Real>,
    ) -> PyResult<PySurfaceModel> {
        let vertices = vertices_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point2::new(a[0], a[1]))
            .collect();

        let faces = faces_py
            .as_array()
            .axis_iter(Axis(0))
            .map(|a| Point3::new(a[0], a[1], a[2]))
            .collect();

        let z = z_py.as_array().to_vec();

        Ok(PySurfaceModel {
            inner: Some(SurfaceModel::new(vertices, faces, z)),
        })
    }

    #[pymethods]
    impl PySurfaceModel {
        /// Query the surface at (x, y) to get the interpolated value.
        pub fn query(&self, x: Real, y: Real) -> PyResult<Option<Real>> {
            let inner = self
                .inner
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("SurfaceModel has been consumed"))?;
            Ok(inner.query(Point2::new(x, y)))
        }

        /// Query the surface for many points at once, returning a 1D float array.
        ///
        /// `xy` is an `(N, 2)` float array with columns `[x, y]`.
        ///
        /// Returns an `(N,)` float array. Points outside the surface are set to 0.0.
        pub fn query_many<'py>(
            &self,
            py: Python<'py>,
            xy: PyReadonlyArray2<Real>,
        ) -> PyResult<Bound<'py, PyArray1<Real>>> {
            let inner = self
                .inner
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("SurfaceModel has been consumed"))?;

            // Capture the input view
            let coords = xy.as_array();
            let n = coords.nrows();

            // Initialise output buffer
            let mut buf = Array1::<Real>::zeros(n);

            // Detach from GIL to allow multi-threaded Python to keep moving
            // (or to allow Rayon par_iter if you decide to add it later)
            py.detach(|| {
                azip!((out in &mut buf, row in coords.rows()) {
                    let pt = Point2::new(row[0], row[1]);
                    if let Some(val) = inner.query(pt) {
                        *out = val;
                    }
                });
            });

            Ok(buf.into_pyarray(py))
        }

        /// Return a serialisable Python dict describing this surface.
        pub fn view<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
            let inner = self
                .inner
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("SurfaceModel has been consumed"))?;
            pythonize(py, &inner.view()).map_err(|e| e.into())
        }
    }

    /// Vectorised Porter-Duff "over" blend of two ``(N, 6)`` quality arrays.
    ///
    /// Each row of *lhs* (foreground) is composited over the corresponding row
    /// of *rhs* (background) using the same ``Quality::blend`` formula used
    /// internally by the BVH query loop.  Column order is
    /// ``[rho, vp, vs, qp, qs, alpha]``.
    ///
    /// Called from ``nzcvm.qualities.blend`` via ``xr.apply_ufunc`` so that
    /// Python / xarray orchestrate the spatial dimensions while the hot loop
    /// runs in Rust with the GIL released.
    #[pyfunction]
    pub fn blend_many<'py>(
        py: Python<'py>,
        lhs: PyReadonlyArray2<Real>,
        rhs: PyReadonlyArray2<Real>,
    ) -> PyResult<Bound<'py, PyArray2<Real>>> {
        let la = lhs.as_array();
        let ra = rhs.as_array();
        let n = la.nrows();
        if ra.nrows() != n {
            return Err(PyValueError::new_err(format!(
                "lhs and rhs must have the same number of rows; got lhs={n}, rhs={}",
                ra.nrows(),
            )));
        }
        if la.ncols() != 6 || ra.ncols() != 6 {
            return Err(PyValueError::new_err(
                "lhs and rhs must each have exactly 6 columns [rho, vp, vs, qp, qs, alpha]",
            ));
        }
        let mut out = Array2::<Real>::zeros((n, 6));
        py.detach(|| {
            azip!((mut o in out.rows_mut(), l in la.rows(), r in ra.rows()) {
                let bq = Quality::from(l).blend(&Quality::from(r));
                o[0] = bq.rho;
                o[1] = bq.vp;
                o[2] = bq.vs;
                o[3] = bq.qp;
                o[4] = bq.qs;
                o[5] = bq.alpha;
            });
        });
        Ok(out.into_pyarray(py))
    }

    /// Python-facing coastline for signed-distance queries.
    #[pyclass]
    pub struct PyCoastline {
        inner: Coastline,
    }

    /// Build a [`PyCoastline`] from an array of segments.
    ///
    /// # Arguments
    /// * `segments_py` – `(S, 2, 2)` array of segment endpoints, indexed
    ///   `[segment, endpoint, (x, y)]`.  The segments are expected to close
    ///   into one or more rings.
    #[pyfunction]
    pub fn coastline(segments_py: PyReadonlyArray3<Real>) -> PyResult<PyCoastline> {
        let array = segments_py.as_array();
        let shape = array.shape();
        if shape[1] != 2 || shape[2] != 2 {
            return Err(PyValueError::new_err(format!(
                "segments must have shape (S, 2, 2), got ({}, {}, {})",
                shape[0], shape[1], shape[2]
            )));
        }
        let segments = array
            .axis_iter(Axis(0))
            .map(|s| {
                Segment::new(
                    Point2::new(s[[0, 0]], s[[0, 1]]),
                    Point2::new(s[[1, 0]], s[[1, 1]]),
                )
            })
            .collect();
        Ok(PyCoastline {
            inner: Coastline::new(segments),
        })
    }

    #[pymethods]
    impl PyCoastline {
        /// Number of segments in the coastline.
        pub fn __len__(&self) -> usize {
            self.inner.len()
        }

        /// Signed distance from each point to the coastline, negative inside.
        ///
        /// Parameters
        /// ----------
        /// x, y :
        ///     Two 1-D float arrays of the same length.
        ///
        /// Returns
        /// -------
        /// numpy.ndarray
        ///     Distances, negative where the point lies inside the coastline.
        pub fn signed_distance<'py>(
            &self,
            py: Python<'py>,
            x: PyReadonlyArray1<Real>,
            y: PyReadonlyArray1<Real>,
        ) -> PyResult<Bound<'py, PyArray1<Real>>> {
            let x = x.as_slice()?;
            let y = y.as_slice()?;
            if x.len() != y.len() {
                return Err(PyValueError::new_err(format!(
                    "x and y must be the same length, got {} and {}",
                    x.len(),
                    y.len()
                )));
            }
            let mut out = Array1::<Real>::zeros(x.len());
            py.detach(|| {
                self.inner
                    .signed_distance_many(x, y, out.as_slice_mut().expect("contiguous"))
            });
            Ok(out.into_pyarray(py))
        }
    }

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyMeshModel>()?;
        m.add_class::<PySurfaceModel>()?;
        m.add_class::<PyCoastline>()?;
        m.add_class::<PyModelTree>()?;
        m.add_class::<QueryParams>()?;
        m.add_class::<QueryCoordinates>()?;
        m.add_function(wrap_pyfunction!(mesh_model, m)?)?;
        m.add_function(wrap_pyfunction!(mesh_model_open, m)?)?;
        m.add_function(wrap_pyfunction!(index_fingerprint, m)?)?;
        m.add_function(wrap_pyfunction!(surface_model, m)?)?;
        m.add_function(wrap_pyfunction!(coastline, m)?)?;
        m.add_function(wrap_pyfunction!(model_tree, m)?)?;
        m.add_function(wrap_pyfunction!(blend_many, m)?)?;

        Ok(())
    }
}
