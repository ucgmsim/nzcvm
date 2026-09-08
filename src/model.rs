use crate::quality::{Quality, barycentric_interpolate};
use crate::real::Real;
use crate::simplex::Simplex;
use deepsize::{Context, DeepSizeOf};
use enum_dispatch::enum_dispatch;
use nalgebra::{Point3, Point4};

/// A type that can report the seismic quality at a point inside a simplex.
#[enum_dispatch]
pub trait Queryable {
    /// Return the quality at `point` inside `simplex`, looking up vertex
    /// properties from the qualities slice.
    fn quality_at(&self, qualities: &[Quality], simplex: &Simplex, point: &Point3<Real>)
    -> Quality;
}

/// Per-simplex model variant: either constant or barycentric interpolation.
#[enum_dispatch(Queryable)]
pub enum Model {
    Constant(ConstantModel),
    Interpolate(InterpolateModel),
}

impl DeepSizeOf for Model {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        0
    }
}

/// Model that returns the same quality regardless of position within the simplex.
pub struct ConstantModel {
    /// Index into the qualities array.
    pub quality: u32,
}

impl Queryable for ConstantModel {
    fn quality_at(
        &self,
        qualities: &[Quality],
        _simplex: &Simplex,
        _point: &Point3<Real>,
    ) -> Quality {
        qualities[self.quality as usize]
    }
}

/// Model that interpolates quality using barycentric coordinates within the simplex.
pub struct InterpolateModel {
    /// Indices of the four vertex qualities, stored in `(x, y, z, w)` order
    /// matching the simplex vertices.
    pub qualities: Point4<u32>,
}

fn interpolate_quality(
    indices: &Point4<u32>,
    qualities: &[Quality],
    simplex: &Simplex,
    point: &Point3<Real>,
) -> Quality {
    let bary = simplex.barycentric_coordinates(*point);
    let q0 = qualities[indices.w as usize];
    let q1 = qualities[indices.x as usize];
    let q2 = qualities[indices.y as usize];
    let q3 = qualities[indices.z as usize];
    barycentric_interpolate([q0, q1, q2, q3], [bary.w, bary.x, bary.y, bary.z])
}

impl Queryable for InterpolateModel {
    fn quality_at(
        &self,
        qualities: &[Quality],
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        interpolate_quality(&self.qualities, qualities, simplex, point)
    }
}

/// Per-mesh map from simplex index to its model.
///
/// Meshes are almost always homogeneous (basins are all-interpolate), so the
/// common cases store just the quality indices in a flat array — 16 bytes per
/// simplex for interpolation, 4 for constant — instead of a `Vec<Model>` that
/// pays an enum tag per element.  Heterogeneous meshes fall back to `Mixed`.
pub enum ModelMap {
    Interpolate(Vec<Point4<u32>>),
    Constant(Vec<u32>),
    Mixed(Vec<Model>),
}

impl ModelMap {
    /// Build a map from a list of per-simplex models, collapsing to a
    /// homogeneous representation when possible.
    pub fn from_models(models: Vec<Model>) -> Self {
        if models.iter().all(|m| matches!(m, Model::Interpolate(_))) {
            ModelMap::Interpolate(
                models
                    .into_iter()
                    .map(|m| match m {
                        Model::Interpolate(im) => im.qualities,
                        Model::Constant(_) => unreachable!(),
                    })
                    .collect(),
            )
        } else if models.iter().all(|m| matches!(m, Model::Constant(_))) {
            ModelMap::Constant(
                models
                    .into_iter()
                    .map(|m| match m {
                        Model::Constant(cm) => cm.quality,
                        Model::Interpolate(_) => unreachable!(),
                    })
                    .collect(),
            )
        } else {
            ModelMap::Mixed(models)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ModelMap::Interpolate(v) => v.len(),
            ModelMap::Constant(v) => v.len(),
            ModelMap::Mixed(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gather the map into a new order: entry `i` of the result is entry
    /// `order[i]` of `self`.  Used to match the BVH's leaf ordering.
    pub(crate) fn reorder(self, order: &[u32]) -> Self {
        fn gather<T: Copy>(v: Vec<T>, order: &[u32]) -> Vec<T> {
            order.iter().map(|&i| v[i as usize]).collect()
        }
        match self {
            ModelMap::Interpolate(v) => ModelMap::Interpolate(gather(v, order)),
            ModelMap::Constant(v) => ModelMap::Constant(gather(v, order)),
            ModelMap::Mixed(v) => {
                let mut slots: Vec<Option<Model>> = v.into_iter().map(Some).collect();
                ModelMap::Mixed(
                    order
                        .iter()
                        .map(|&i| slots[i as usize].take().expect("duplicate index in order"))
                        .collect(),
                )
            }
        }
    }

    /// Return the quality at `point` inside simplex `index`.
    pub fn quality_at(
        &self,
        index: usize,
        qualities: &[Quality],
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        match self {
            ModelMap::Interpolate(v) => interpolate_quality(&v[index], qualities, simplex, point),
            ModelMap::Constant(v) => qualities[v[index] as usize],
            ModelMap::Mixed(v) => v[index].quality_at(qualities, simplex, point),
        }
    }
}

impl DeepSizeOf for ModelMap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            ModelMap::Interpolate(v) => v.capacity() * std::mem::size_of::<Point4<u32>>(),
            ModelMap::Constant(v) => v.capacity() * std::mem::size_of::<u32>(),
            ModelMap::Mixed(v) => v.deep_size_of_children(context),
        }
    }
}
