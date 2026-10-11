//! Error-bounded triangulated irregular networks (TINs) of gridded surfaces.
//!
//! [`adaptive_tin`] triangulates a polygon so that the piecewise-linear
//! interpolant of every input surface stays within that surface's vertical
//! tolerance at each of its sample points, inside the polygon and along its
//! boundary.  It is the greedy insertion algorithm of Garland & Heckbert,
//! *Fast Polygonal Approximation of Terrains and Height Fields* (1995),
//! applied to several surfaces at once and run in passes: each pass inserts
//! the worst sample of every triangle and boundary edge that misses a
//! tolerance, until none does.
//!
//! Vertices are drawn from the samples themselves, so the TIN reproduces each
//! surface exactly where it adds a vertex and spends triangles only where a
//! surface bends.  A planar slope, however steep, needs none.

use crate::barycentric::{self, triangle_coordinates};
use crate::grid::StructuredGrid;
use spade::handles::{FixedFaceHandle, FixedVertexHandle, InnerTag};
use spade::{ConstrainedDelaunayTriangulation, Point2, PositionInTriangulation, Triangulation};
use std::collections::VecDeque;

/// One surface for [`adaptive_tin`] to approximate.
pub struct TinSurface<'a> {
    /// The surface, interpolated at vertices and samples.
    pub grid: &'a StructuredGrid,
    /// Points inside the polygon the TIN must fit, with the surface's value
    /// at each: for example the grid nodes and finer points within its cells.
    pub samples: Vec<(Point2<f64>, f64)>,
    /// Spacing of the points along each boundary edge at which the TIN must
    /// also fit the surface.
    pub boundary_spacing: f64,
    /// Largest vertical error the TIN may leave at a sample.
    pub tolerance: f64,
}

impl<'a> TinSurface<'a> {
    /// Sample `grid` at `points`, and every `boundary_spacing` along the
    /// boundary.
    pub fn new(
        grid: &'a StructuredGrid,
        points: &[Point2<f64>],
        boundary_spacing: f64,
        tolerance: f64,
    ) -> Self {
        let samples = points.iter().map(|&p| (p, evaluate(grid, p))).collect();
        Self {
            grid,
            samples,
            boundary_spacing,
            tolerance,
        }
    }
}

/// A triangulation: vertex positions and counter-clockwise vertex-index triples.
pub struct Tin {
    pub vertices: Vec<Point2<f64>>,
    pub triangles: Vec<[usize; 3]>,
}

/// Reasons [`adaptive_tin`] can fail.
#[derive(Debug, PartialEq)]
pub enum TinError {
    /// The boundary crosses itself or has fewer than three distinct
    /// vertices, or the triangulation rejected a vertex (e.g. a non-finite
    /// coordinate).
    InvalidBoundary,
    /// A tolerance, boundary spacing or `max_edge` is not positive.
    InvalidParameter,
}

impl std::fmt::Display for TinError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidBoundary => {
                "boundary must be a simple polygon with at least three vertices"
            }
            Self::InvalidParameter => "tolerances, boundary spacings and max_edge must be positive",
        })
    }
}

type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

/// Distance, relative to the boundary's extent, within which a new point is
/// taken to be the vertex already there.  A long-edge midpoint can land a
/// rounding error from a sample, and two vertices that close would leave a
/// sliver triangle that collapses in single precision.
const SNAP: f64 = 1e-9;

/// Triangulate the polygon `boundary` to fit every surface within its
/// tolerance.
///
/// `boundary` lists the polygon's vertices in order, without repeating the
/// first; it must be simple.  Its edges are constraints, so no triangle
/// crosses it; the TIN splits them where a surface needs it.  With
/// `max_edge`, triangles are also split along their longest edge until no
/// edge is longer.  Only triangles inside the polygon are returned.
pub fn adaptive_tin(
    boundary: &[Point2<f64>],
    surfaces: &[TinSurface],
    max_edge: Option<f64>,
) -> Result<Tin, TinError> {
    // Written to reject NaN as well as non-positive values.
    let positive = |v: f64| v > 0.0;
    if !surfaces
        .iter()
        .all(|s| positive(s.tolerance) && positive(s.boundary_spacing))
        || max_edge.is_some_and(|m| !positive(m))
    {
        return Err(TinError::InvalidParameter);
    }
    let n = boundary.len();
    let edges = (0..n).map(|i| [i, (i + 1) % n]).collect();
    let mut self_intersecting = false;
    let mut cdt = Cdt::try_bulk_load_cdt(boundary.to_vec(), edges, |_| self_intersecting = true)
        .map_err(|_| TinError::InvalidBoundary)?;
    if self_intersecting || cdt.num_inner_faces() == 0 {
        return Err(TinError::InvalidBoundary);
    }

    // values[s][v]: surface s at vertex v.  Vertex handles are dense indices.
    let mut values: Vec<Vec<f64>> = surfaces
        .iter()
        .map(|s| {
            cdt.vertices()
                .map(|v| evaluate(s.grid, v.position()))
                .collect()
        })
        .collect();

    let (min, max) = boundary.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(min, max), p| {
            (
                [min[0].min(p.x), min[1].min(p.y)],
                [max[0].max(p.x), max[1].max(p.y)],
            )
        },
    );
    let snap = SNAP * (max[0] - min[0]).hypot(max[1] - min[1]);

    let mut known_vertices = cdt.num_vertices();
    let mut settled: Vec<Vec<Settled>> = surfaces
        .iter()
        .map(|s| vec![Settled::Unchecked; s.samples.len()])
        .collect();
    loop {
        let mut inserts = worst_samples(&cdt, surfaces, &values, &mut settled);
        inserts.extend(boundary_splits(&cdt, surfaces, &values));
        if let Some(max_edge) = max_edge {
            inserts.extend(long_edge_splits(&cdt, max_edge, &inserts));
        }
        let vertices_before = cdt.num_vertices();
        for insert in inserts {
            let Some(handle) = insert_vertex(&mut cdt, &insert, snap)? else {
                continue;
            };
            if handle.index() < known_vertices {
                // Already a vertex (two faces split the same edge, or the
                // point snapped to one).
                continue;
            }
            known_vertices += 1;
            for (s, surface) in surfaces.iter().enumerate() {
                let value = match insert.sample {
                    Some((sample_surface, value)) if sample_surface == s => value,
                    _ => evaluate(surface.grid, insert.position),
                };
                values[s].push(value);
            }
        }
        // Every insert adds a vertex unless rounding put it on an existing
        // one, so a pass that adds none would repeat forever.
        if cdt.num_vertices() == vertices_before {
            break;
        }
    }

    let inside = inside_faces(&cdt);
    let triangles = cdt
        .inner_faces()
        .filter(|f| inside[f.index()])
        .map(|f| f.vertices().map(|v| v.fix().index()))
        .collect();
    let vertices = cdt.vertices().map(|v| v.position()).collect();
    Ok(Tin {
        vertices,
        triangles,
    })
}

/// A vertex to add: a surface sample, or a point with no sample behind it.
struct Insert {
    /// The face the point splits, or a face beside the boundary edge it
    /// splits.
    face: FixedFaceHandle<InnerTag>,
    /// A vertex near the point, to start locating it from.
    hint: FixedVertexHandle,
    position: Point2<f64>,
    /// The surface the point samples, and its value there.
    sample: Option<(usize, f64)>,
    /// The ends of the boundary edge the point splits, if it is on one.
    boundary_edge: Option<[FixedVertexHandle; 2]>,
}

/// Add `insert`'s point to the triangulation, returning its vertex, or
/// `None` if an earlier insert in the pass already split its boundary edge.
/// A point within `snap` of a vertex returns that vertex instead.
fn insert_vertex(
    cdt: &mut Cdt,
    insert: &Insert,
    snap: f64,
) -> Result<Option<FixedVertexHandle>, TinError> {
    if let Some(v) = vertex_near(cdt, insert.position, insert.hint, snap) {
        return Ok(Some(v));
    }
    let Some([a, b]) = insert.boundary_edge else {
        // Inserts come in face order, which jumps about the map, so start
        // the point location next to the point rather than at the last
        // insert.
        return cdt
            .insert_with_hint(insert.position, insert.hint)
            .map(Some)
            .map_err(|_| TinError::InvalidBoundary);
    };
    let Some(edge) = cdt.get_edge_from_neighbors(a, b) else {
        return Ok(None);
    };
    let edge = edge.fix().as_undirected();
    if !cdt.is_constraint_edge(edge) {
        return Ok(None);
    }
    // The point lies on the edge only to within rounding, so inserting it
    // need not split the constraint.  Lift the constraint, insert the point,
    // and constrain both halves.
    cdt.remove_constraint_edge(edge);
    let v = cdt
        .insert_with_hint(insert.position, a)
        .map_err(|_| TinError::InvalidBoundary)?;
    for (from, to) in [(a, v), (v, b)] {
        if from != to
            && !cdt.exists_constraint(from, to)
            && cdt.try_add_constraint(from, to).is_empty()
        {
            return Err(TinError::InvalidBoundary);
        }
    }
    Ok(Some(v))
}

/// What the previous pass learned about a sample.
#[derive(Clone, Copy)]
enum Settled {
    /// Not yet checked, or its triangle had to be split.
    Unchecked,
    /// Within tolerance in this triangle, which had these vertices.  While
    /// the face still has them the interpolant there is unchanged.
    Within(FixedFaceHandle<InnerTag>, [FixedVertexHandle; 3]),
    /// A vertex, or outside the triangulation: never needs a vertex.
    Done,
}

/// The error of `interpolated` against a sample's `value`, relative to the
/// tolerance.  A missing sample value never forces a vertex, but a missing
/// vertex value (an interpolant of NaN) at a valid sample counts as too far
/// off, so the TIN keeps refining around it.
fn relative_error(interpolated: f64, value: f64, tolerance: f64) -> f64 {
    if value.is_nan() {
        0.0
    } else if interpolated.is_nan() {
        f64::INFINITY
    } else {
        (interpolated - value).abs() / tolerance
    }
}

/// Find, for each face, the sample with the largest tolerance-relative
/// error, and return those whose error exceeds the tolerance.
///
/// Only samples whose triangle changed since the last pass are checked:
/// insertion replaces every triangle it touches, so a sample whose triangle
/// survives is still within tolerance.
fn worst_samples(
    cdt: &Cdt,
    surfaces: &[TinSurface],
    values: &[Vec<f64>],
    settled: &mut [Vec<Settled>],
) -> Vec<Insert> {
    // (relative error, surface, sample) of the worst sample seen in each face.
    let mut worst: Vec<(f64, usize, usize)> = vec![(1.0, usize::MAX, 0); cdt.num_all_faces()];
    let mut hint = FixedVertexHandle::from_index(0);
    for (s, surface) in surfaces.iter().enumerate() {
        for (j, &(position, value)) in surface.samples.iter().enumerate() {
            match settled[s][j] {
                Settled::Done => continue,
                Settled::Within(face, vertices) => {
                    if cdt.face(face).vertices().map(|v| v.fix()) == vertices {
                        continue;
                    }
                    hint = vertices[0];
                }
                Settled::Unchecked => {}
            }
            let Some(face) = containing_face(cdt, position, hint) else {
                settled[s][j] = Settled::Done;
                continue;
            };
            let face = cdt.face(face);
            let vertices = face.vertices();
            hint = vertices[0].fix();
            let interpolated = linear_interpolate(
                vertices.map(|v| v.position()),
                vertices.map(|v| values[s][v.fix().index()]),
                position,
            );
            let error = relative_error(interpolated, value, surface.tolerance);
            if error > 1.0 {
                settled[s][j] = Settled::Unchecked;
                let slot = &mut worst[face.index()];
                if error > slot.0 {
                    *slot = (error, s, j);
                }
            } else {
                settled[s][j] = Settled::Within(face.fix(), vertices.map(|v| v.fix()));
            }
        }
    }
    cdt.inner_faces()
        .filter_map(|face| {
            let (_, s, j) = worst[face.index()];
            (s != usize::MAX).then(|| {
                let (position, value) = surfaces[s].samples[j];
                Insert {
                    face: face.fix(),
                    hint: face.vertices()[0].fix(),
                    position,
                    sample: Some((s, value)),
                    boundary_edge: None,
                }
            })
        })
        .collect()
}

/// Find, for each boundary edge, the point every `boundary_spacing` along
/// it with the largest tolerance-relative error, and return those whose
/// error exceeds the tolerance.
fn boundary_splits(cdt: &Cdt, surfaces: &[TinSurface], values: &[Vec<f64>]) -> Vec<Insert> {
    cdt.undirected_edges()
        .filter(|edge| cdt.is_constraint_edge(edge.fix()))
        .filter_map(|edge| {
            let [a, b] = edge.vertices();
            let [pa, pb] = [a.position(), b.position()];
            let length = ((pb.x - pa.x).powi(2) + (pb.y - pa.y).powi(2)).sqrt();
            let mut worst = (1.0, None);
            for (s, surface) in surfaces.iter().enumerate() {
                let [va, vb] = [a, b].map(|v| values[s][v.fix().index()]);
                let pieces = (length / surface.boundary_spacing).ceil() as usize;
                for k in 1..pieces {
                    let t = k as f64 / pieces as f64;
                    let p = Point2::new(pa.x + t * (pb.x - pa.x), pa.y + t * (pb.y - pa.y));
                    let value = evaluate(surface.grid, p);
                    let error = relative_error(va + t * (vb - va), value, surface.tolerance);
                    if error > worst.0 {
                        worst = (error, Some((s, p, value)));
                    }
                }
            }
            let (s, position, value) = worst.1?;
            let directed = edge.as_directed();
            let face = directed
                .face()
                .as_inner()
                .or(directed.rev().face().as_inner())?;
            Some(Insert {
                face: face.fix(),
                hint: a.fix(),
                position,
                sample: Some((s, value)),
                boundary_edge: Some([a.fix(), b.fix()]),
            })
        })
        .collect()
}

/// A vertex within `snap` of `position`, if one is a corner of the face or
/// edge containing it.
fn vertex_near(
    cdt: &Cdt,
    position: Point2<f64>,
    hint: FixedVertexHandle,
    snap: f64,
) -> Option<FixedVertexHandle> {
    let near = |v: spade::handles::VertexHandle<_, _, _, _>| {
        (v.position().distance_2(position) <= snap * snap).then(|| v.fix())
    };
    match cdt.locate_with_hint(position, hint) {
        PositionInTriangulation::OnVertex(v) => Some(v),
        PositionInTriangulation::OnEdge(edge) => cdt
            .directed_edge(edge)
            .vertices()
            .into_iter()
            .find_map(near),
        PositionInTriangulation::OnFace(face) => {
            cdt.face(face).vertices().into_iter().find_map(near)
        }
        PositionInTriangulation::OutsideOfConvexHull(_)
        | PositionInTriangulation::NoTriangulation => None,
    }
}

/// The inner face containing `position`, or `None` if it is a vertex or
/// outside the triangulation.  A point on an edge takes either adjacent
/// inner face; the interpolant agrees on the shared edge.
fn containing_face(
    cdt: &Cdt,
    position: Point2<f64>,
    hint: FixedVertexHandle,
) -> Option<FixedFaceHandle<InnerTag>> {
    match cdt.locate_with_hint(position, hint) {
        PositionInTriangulation::OnFace(face) => Some(face),
        PositionInTriangulation::OnEdge(edge) => {
            let edge = cdt.directed_edge(edge);
            edge.face()
                .as_inner()
                .or(edge.rev().face().as_inner())
                .map(|f| f.fix())
        }
        PositionInTriangulation::OnVertex(_)
        | PositionInTriangulation::OutsideOfConvexHull(_)
        | PositionInTriangulation::NoTriangulation => None,
    }
}

/// Linear interpolation of `values` at the corners of a triangle.
fn linear_interpolate(corners: [Point2<f64>; 3], values: [f64; 3], p: Point2<f64>) -> f64 {
    let point = |q: Point2<f64>| nalgebra::Point2::new(q.x, q.y);
    barycentric::interpolate(values, triangle_coordinates(corners.map(point), point(p)))
}

/// The surface at `p`, extending the edge of its grid outside it.
fn evaluate(grid: &StructuredGrid, p: Point2<f64>) -> f64 {
    grid.query(p.x, p.y, true).0
}

/// Which inner faces lie inside the polygon, indexed by face index.
///
/// A face is inside when reaching it from the outer face crosses an odd
/// number of constraint edges.
fn inside_faces(cdt: &Cdt) -> Vec<bool> {
    let mut crossings: Vec<Option<usize>> = vec![None; cdt.num_all_faces()];
    let mut queue = VecDeque::new();
    let mut visit = |face: FixedFaceHandle<InnerTag>, depth: usize, queue: &mut VecDeque<_>| {
        if crossings[face.index()].is_none() {
            crossings[face.index()] = Some(depth);
            queue.push_back((face, depth));
        }
    };
    // Faces on the convex hull border the outer face.
    for edge in cdt.convex_hull() {
        let inner = if edge.face().is_outer() {
            edge.rev()
        } else {
            edge
        };
        let inner = inner
            .face()
            .as_inner()
            .expect("one side of a hull edge is inner");
        let crossed = cdt.is_constraint_edge(edge.fix().as_undirected());
        visit(inner.fix(), usize::from(crossed), &mut queue);
    }
    while let Some((face, depth)) = queue.pop_front() {
        for edge in cdt.face(face).adjacent_edges() {
            if let Some(neighbour) = edge.rev().face().as_inner() {
                let crossed = cdt.is_constraint_edge(edge.fix().as_undirected());
                visit(neighbour.fix(), depth + usize::from(crossed), &mut queue);
            }
        }
    }
    crossings
        .iter()
        .map(|c| c.is_some_and(|d| d % 2 == 1))
        .collect()
}

/// Midpoints of the longest edge of each inside face, other than those
/// `inserts` already splits, whose longest edge exceeds `max_edge`.
fn long_edge_splits(cdt: &Cdt, max_edge: f64, inserts: &[Insert]) -> Vec<Insert> {
    let inside = inside_faces(cdt);
    let mut splitting = vec![false; cdt.num_all_faces()];
    inserts
        .iter()
        .for_each(|i| splitting[i.face.index()] = true);
    cdt.inner_faces()
        .filter(|face| inside[face.index()] && !splitting[face.index()])
        .filter_map(|face| {
            let longest = face
                .adjacent_edges()
                .into_iter()
                .max_by(|a, b| a.length_2().total_cmp(&b.length_2()))
                .expect("a face has three edges");
            (longest.length_2() > max_edge * max_edge).then(|| {
                let [a, b] = longest.positions();
                let ends = [longest.from().fix(), longest.to().fix()];
                let on_boundary = cdt.is_constraint_edge(longest.fix().as_undirected());
                Insert {
                    face: face.fix(),
                    hint: ends[0],
                    position: Point2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y)),
                    sample: None,
                    boundary_edge: on_boundary.then_some(ends),
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use proptest::prelude::*;

    /// A `(n, n)` grid over `[0, side]²` sampling `f`, with every node a
    /// sample.  The grid is leaked, which is fine for a test.
    fn surface(
        n: usize,
        side: f64,
        tolerance: f64,
        f: impl Fn(f64, f64) -> f64,
    ) -> TinSurface<'static> {
        let coordinate = |i: usize| side * i as f64 / (n - 1) as f64;
        let x = Array2::from_shape_fn((n, n), |(_, k)| coordinate(k));
        let y = Array2::from_shape_fn((n, n), |(j, _)| coordinate(j));
        let values = Array2::from_shape_fn((n, n), |(j, k)| f(coordinate(k), coordinate(j)));
        let points: Vec<_> = x.iter().zip(&y).map(|(&x, &y)| Point2::new(x, y)).collect();
        let grid = Box::leak(Box::new(StructuredGrid::new(x, y, values).unwrap()));
        TinSurface::new(grid, &points, side / (n - 1) as f64, tolerance)
    }

    fn square(side: f64) -> Vec<Point2<f64>> {
        vec![
            Point2::new(0.0, 0.0),
            Point2::new(side, 0.0),
            Point2::new(side, side),
            Point2::new(0.0, side),
        ]
    }

    /// The largest error of the TIN's interpolant at the surface's samples.
    fn max_error(tin: &Tin, surface: &TinSurface) -> f64 {
        let point = |q: Point2<f64>| nalgebra::Point2::new(q.x, q.y);
        let vertex_values: Vec<f64> = tin
            .vertices
            .iter()
            .map(|&p| evaluate(surface.grid, p))
            .collect();
        surface
            .samples
            .iter()
            .map(|&(p, value)| {
                // The triangle the sample is deepest inside, whose smallest
                // coordinate is about zero for a sample on an edge.
                let (triangle, weights) = tin
                    .triangles
                    .iter()
                    .map(|t| {
                        let corners = t.map(|i| point(tin.vertices[i]));
                        (t, triangle_coordinates(corners, point(p)))
                    })
                    .max_by(|(_, a), (_, b)| {
                        let min = |w: &[f64; 3]| w.iter().copied().fold(f64::INFINITY, f64::min);
                        min(a).total_cmp(&min(b))
                    })
                    .expect("a TIN has a triangle");
                assert!(
                    weights.iter().all(|&w| w >= -1e-9),
                    "sample {p:?} outside the TIN"
                );
                let interpolated =
                    barycentric::interpolate(triangle.map(|i| vertex_values[i]), weights);
                (interpolated - value).abs()
            })
            .fold(0.0, f64::max)
    }

    /// A smooth random surface: a plane, a sinusoid and a Gaussian bump.
    #[derive(Clone, Debug)]
    struct Terrain {
        slope: (f64, f64),
        wave: (f64, f64, f64, f64),
        bump: (f64, f64, f64, f64),
    }

    impl Terrain {
        fn at(&self, x: f64, y: f64) -> f64 {
            let (amplitude, kx, ky, phase) = self.wave;
            let (height, cx, cy, width) = self.bump;
            let r2 = (x - cx).powi(2) + (y - cy).powi(2);
            self.slope.0 * x
                + self.slope.1 * y
                + amplitude * (kx * x + ky * y + phase).sin()
                + height * (-r2 / (width * width)).exp()
        }
    }

    fn terrain(side: f64) -> impl Strategy<Value = Terrain> {
        let k = 0.5..10.0;
        (
            (-2.0..2.0, -2.0..2.0),
            (0.0..20.0, k.clone(), k, 0.0..std::f64::consts::TAU),
            (-50.0..50.0, 0.0..side, 0.0..side, 2.0..side / 2.0),
        )
            .prop_map(move |(slope, (amplitude, kx, ky, phase), bump)| Terrain {
                slope,
                // Wavenumbers in cycles across the square.
                wave: (amplitude, kx / side, ky / side, phase),
                bump,
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// The TIN's guarantee: every surface's interpolant is within its
        /// tolerance at each of its samples, with or without `max_edge`.
        #[test]
        fn prop_every_sample_is_within_tolerance(
            terrains in prop::collection::vec((terrain(100.0), 0.05..5.0, 5usize..30), 1..3),
            max_edge in prop::option::of(10.0..100.0),
        ) {
            let surfaces: Vec<_> = terrains
                .iter()
                .map(|(t, tolerance, n)| surface(*n, 100.0, *tolerance, |x, y| t.at(x, y)))
                .collect();
            let tin = adaptive_tin(&square(100.0), &surfaces, max_edge).unwrap();
            // No near-duplicate vertices leave slivers behind.
            for t in &tin.triangles {
                let [a, b, c] = t.map(|i| tin.vertices[i]);
                let area = 0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y));
                prop_assert!(area > 1e-6, "sliver {:?} of area {area}", [a, b, c]);
            }
            for surface in &surfaces {
                let error = max_error(&tin, surface);
                prop_assert!(
                    error <= surface.tolerance * (1.0 + 1e-9),
                    "error {error} over tolerance {}", surface.tolerance
                );
            }
        }
    }

    #[test]
    fn test_plane_needs_only_the_boundary() {
        let plane = surface(21, 100.0, 0.01, |x, y| 3.0 * x - 2.0 * y + 7.0);
        let tin = adaptive_tin(&square(100.0), &[plane], None).unwrap();
        assert_eq!(tin.vertices.len(), 4);
        assert_eq!(tin.triangles.len(), 2);
    }

    #[test]
    fn test_curved_surface_meets_tolerance() {
        let tolerance = 0.5;
        let bump = surface(41, 100.0, tolerance, |x, y| {
            50.0 * (-((x - 40.0).powi(2) + (y - 60.0).powi(2)) / 400.0).exp()
        });
        let tin = adaptive_tin(&square(100.0), std::slice::from_ref(&bump), None).unwrap();
        assert!(max_error(&tin, &bump) <= tolerance + 1e-9);
        // Far fewer vertices than samples.
        assert!(tin.vertices.len() < bump.samples.len() / 4);
    }

    #[test]
    fn test_every_surface_meets_its_own_tolerance() {
        let surfaces = [
            surface(31, 100.0, 1.0, |x, _| 20.0 * (x / 15.0).sin()),
            surface(17, 100.0, 0.2, |_, y| (y - 50.0).powi(2) / 100.0),
        ];
        let tin = adaptive_tin(&square(100.0), &surfaces, None).unwrap();
        assert!(max_error(&tin, &surfaces[0]) <= 1.0 + 1e-9);
        assert!(max_error(&tin, &surfaces[1]) <= 0.2 + 1e-9);
    }

    #[test]
    fn test_concave_polygon_keeps_only_inside_triangles() {
        // An L shape: the notch [50, 100]² is outside.
        let boundary = vec![
            Point2::new(0.0, 0.0),
            Point2::new(100.0, 0.0),
            Point2::new(100.0, 50.0),
            Point2::new(50.0, 50.0),
            Point2::new(50.0, 100.0),
            Point2::new(0.0, 100.0),
        ];
        let plane = surface(5, 100.0, 1.0, |_, _| 0.0);
        let tin = adaptive_tin(&boundary, &[plane], None).unwrap();
        let area: f64 = tin
            .triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| tin.vertices[i]);
                0.5 * ((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y))
            })
            .sum();
        assert!((area - 7500.0).abs() < 1e-6, "area {area}");
    }

    #[test]
    fn test_max_edge_bounds_every_edge() {
        let plane = surface(3, 100.0, 1.0, |_, _| 0.0);
        let tin = adaptive_tin(&square(100.0), &[plane], Some(30.0)).unwrap();
        for t in &tin.triangles {
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = (tin.vertices[t[i]], tin.vertices[t[j]]);
                let length = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt();
                assert!(length <= 30.0 + 1e-9, "edge {length}");
            }
        }
    }

    #[test]
    fn test_nan_samples_are_ignored() {
        let mut holes = surface(11, 100.0, 0.1, |x, y| (x * y).sqrt());
        holes
            .samples
            .iter_mut()
            .step_by(3)
            .for_each(|n| n.1 = f64::NAN);
        assert!(adaptive_tin(&square(100.0), &[holes], None).is_ok());
    }

    #[test]
    fn test_rejects_bad_parameters() {
        let plane = || surface(3, 1.0, 1.0, |_, _| 0.0);
        let mut zero_tolerance = plane();
        zero_tolerance.tolerance = 0.0;
        assert_eq!(
            adaptive_tin(&square(1.0), &[zero_tolerance], None).err(),
            Some(TinError::InvalidParameter)
        );
        assert_eq!(
            adaptive_tin(&square(1.0), &[plane()], Some(0.0)).err(),
            Some(TinError::InvalidParameter)
        );
        assert_eq!(
            adaptive_tin(&square(1.0)[..2], &[plane()], None).err(),
            Some(TinError::InvalidBoundary)
        );
        // A bow tie crosses itself.
        let mut bow_tie = square(1.0);
        bow_tie.swap(2, 3);
        assert_eq!(
            adaptive_tin(&bow_tie, &[plane()], None).err(),
            Some(TinError::InvalidBoundary)
        );
    }

    /// The largest error of the TIN's interpolant along its boundary, every
    /// `step` along each edge of the square `[0, side]²`.
    fn max_boundary_error(tin: &Tin, surface: &TinSurface, side: f64, step: f64) -> f64 {
        let n = (side / step) as usize;
        let points = (0..=n).flat_map(|k| {
            let t = k as f64 * step;
            [(t, 0.0), (side, t), (t, side), (0.0, t)].map(|(x, y)| Point2::new(x, y))
        });
        let samples: Vec<_> = points.map(|p| (p, evaluate(surface.grid, p))).collect();
        let on_boundary = TinSurface {
            grid: surface.grid,
            samples,
            boundary_spacing: surface.boundary_spacing,
            tolerance: surface.tolerance,
        };
        max_error(tin, &on_boundary)
    }

    #[test]
    fn test_boundary_meets_tolerance_without_interior_samples() {
        // The surface varies only along the boundary, where no interior
        // sample can see it.
        let mut wavy = surface(41, 100.0, 0.5, |x, y| {
            20.0 * (x / 8.0).sin() * (y / 8.0).cos()
        });
        wavy.samples.clear();
        let tin = adaptive_tin(&square(100.0), std::slice::from_ref(&wavy), None).unwrap();
        assert!(tin.vertices.len() > 4, "the boundary must be split");
        assert!(max_boundary_error(&tin, &wavy, 100.0, wavy.boundary_spacing) <= 0.5 + 1e-9);
    }

    #[test]
    fn test_max_edge_terminates_on_exact_multiples() {
        // Boundary edges three times max_edge, whose split pieces round to
        // either side of it.
        for side in [90.0, 0.3, 3e6] {
            let plane = surface(3, side, 1.0, |_, _| 0.0);
            let max_edge = side / 3.0;
            let tin = adaptive_tin(&square(side), &[plane], Some(max_edge)).unwrap();
            assert!(tin.vertices.len() >= 12, "side {side}");
        }
    }

    #[test]
    fn test_missing_vertex_values_still_refine() {
        // NaN at the corner (0, 0), which is a boundary vertex.
        let bump = surface(21, 100.0, 0.5, |x, y| {
            if x == 0.0 && y == 0.0 {
                f64::NAN
            } else {
                50.0 * (-((x - 40.0).powi(2) + (y - 60.0).powi(2)) / 400.0).exp()
            }
        });
        let tin = adaptive_tin(&square(100.0), std::slice::from_ref(&bump), None).unwrap();
        // Every valid sample is either a vertex or fitted by a triangle with
        // no missing corner.
        assert!(max_error(&tin, &bump) <= 0.5 + 1e-9);
    }
}
