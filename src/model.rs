use crate::quality::{Quality, barycentric_interpolate};
use crate::real::Real;
use crate::simplex::Simplex;
use crate::slab::Slab;
use deepsize::{Context, DeepSizeOf};
use nalgebra::{Point3, Point4};

/// How one simplex reports its quality, as supplied when a mesh is built.
///
/// A [`ModelMap`] stores these compactly; this enum only exists to describe
/// a mesh on the way in.
pub enum Model {
    Constant(ConstantModel),
    Interpolate(InterpolateModel),
}

impl From<ConstantModel> for Model {
    fn from(m: ConstantModel) -> Self {
        Model::Constant(m)
    }
}

impl From<InterpolateModel> for Model {
    fn from(m: InterpolateModel) -> Self {
        Model::Interpolate(m)
    }
}

/// Model that returns the same quality regardless of position within the simplex.
pub struct ConstantModel {
    /// Index into the qualities array.
    pub quality: u32,
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

/// Per-mesh map from simplex index to its model.
///
/// Stores just the quality indices in a flat array instead of a `Vec<Model>`
/// that pays an enum tag per element: four per simplex, or one when every
/// simplex is constant.  A constant simplex among interpolating ones is
/// marked with [`NO_INTERPOLATION`] in the second slot.
///
/// Both variants are [`Slab`]s so that the map can be memory-mapped from an
/// index file as readily as built in memory.
pub enum ModelMap {
    Refs(Slab<Point4<u32>>),
    Constant(Slab<u32>),
}

/// Marks a constant simplex in a [`ModelMap::Refs`] map.
///
/// Sits in the second index slot, where an interpolating simplex has a
/// vertex index.  A mesh holds at most `u32::MAX` qualities, so no vertex
/// index is ever this value.
pub const NO_INTERPOLATION: u32 = u32::MAX;

impl ModelMap {
    /// Build a map from a list of per-simplex models, gathered into `order`:
    /// entry `i` of the map is `models[order[i]]`.  Used to match the BVH's
    /// leaf ordering.
    pub fn from_models(models: &[Model], order: &[u32]) -> Self {
        let ordered = order.iter().map(|&i| &models[i as usize]);
        if models.iter().all(|m| matches!(m, Model::Constant(_))) {
            ModelMap::Constant(
                ordered
                    .map(|m| match m {
                        Model::Constant(cm) => cm.quality,
                        Model::Interpolate(_) => unreachable!(),
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        } else {
            ModelMap::Refs(
                ordered
                    .map(|m| match m {
                        Model::Constant(cm) => Point4::new(cm.quality, NO_INTERPOLATION, 0, 0),
                        Model::Interpolate(im) => im.qualities,
                    })
                    .collect::<Vec<_>>()
                    .into(),
            )
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ModelMap::Refs(v) => v.len(),
            ModelMap::Constant(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            ModelMap::Refs(v) => {
                let refs = &v[index];
                if refs.y == NO_INTERPOLATION {
                    qualities[refs.x as usize]
                } else {
                    interpolate_quality(refs, qualities, simplex, point)
                }
            }
            ModelMap::Constant(v) => qualities[v[index] as usize],
        }
    }
}

impl DeepSizeOf for ModelMap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            ModelMap::Refs(v) => v.deep_size_of_children(context),
            ModelMap::Constant(v) => v.deep_size_of_children(context),
        }
    }
}
