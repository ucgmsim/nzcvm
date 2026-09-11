use crate::quality::{Quality, barycentric_interpolate};
use crate::real::Real;
use crate::simplex::Simplex;
use crate::slab::Slab;
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

/// Quality indices of one simplex's four vertices, in `(x, y, z, w)` order.
pub type VertexRefs = [u32; 4];

fn interpolate_quality(
    indices: &VertexRefs,
    qualities: &[Quality],
    simplex: &Simplex,
    point: &Point3<Real>,
) -> Quality {
    let bary = simplex.barycentric_coordinates(*point);
    let [x, y, z, w] = *indices;
    let q0 = qualities[w as usize];
    let q1 = qualities[x as usize];
    let q2 = qualities[y as usize];
    let q3 = qualities[z as usize];
    barycentric_interpolate([q0, q1, q2, q3], [bary.w, bary.x, bary.y, bary.z])
}

impl Queryable for InterpolateModel {
    fn quality_at(
        &self,
        qualities: &[Quality],
        simplex: &Simplex,
        point: &Point3<Real>,
    ) -> Quality {
        interpolate_quality(&self.qualities.into(), qualities, simplex, point)
    }
}

/// Per-mesh map from simplex index to its model.
///
/// Meshes are almost always homogeneous (basins are all-interpolate), so the
/// common cases store just the quality indices in a flat array — 16 bytes per
/// simplex for interpolation, 4 for constant — instead of a `Vec<Model>` that
/// pays an enum tag per element.  A heterogeneous mesh stores the tag in a
/// parallel [`Slab`] instead.
///
/// Every variant is a [`Slab`] so that the map can be memory-mapped from an
/// index file as readily as built in memory.
pub enum ModelMap {
    Interpolate(Slab<VertexRefs>),
    Constant(Slab<u32>),
    Mixed {
        /// [`MIXED_CONSTANT`] or [`MIXED_INTERPOLATE`] per simplex.
        kinds: Slab<u32>,
        /// Four vertex indices per simplex; a constant simplex uses only the
        /// first.
        refs: Slab<VertexRefs>,
    },
}

/// Tag of a constant simplex in a [`ModelMap::Mixed`] map.
pub const MIXED_CONSTANT: u32 = 0;
/// Tag of an interpolating simplex in a [`ModelMap::Mixed`] map.
pub const MIXED_INTERPOLATE: u32 = 1;

impl ModelMap {
    /// Build a map from a list of per-simplex models, collapsing to a
    /// homogeneous representation when possible.
    pub fn from_models(models: Vec<Model>) -> Self {
        if models.iter().all(|m| matches!(m, Model::Interpolate(_))) {
            ModelMap::Interpolate(Slab::Owned(
                models
                    .into_iter()
                    .map(|m| match m {
                        Model::Interpolate(im) => im.qualities.into(),
                        Model::Constant(_) => unreachable!(),
                    })
                    .collect(),
            ))
        } else if models.iter().all(|m| matches!(m, Model::Constant(_))) {
            ModelMap::Constant(Slab::Owned(
                models
                    .into_iter()
                    .map(|m| match m {
                        Model::Constant(cm) => cm.quality,
                        Model::Interpolate(_) => unreachable!(),
                    })
                    .collect(),
            ))
        } else {
            let (kinds, refs) = models
                .into_iter()
                .map(|m| match m {
                    Model::Constant(cm) => (MIXED_CONSTANT, [cm.quality, 0, 0, 0]),
                    Model::Interpolate(im) => (MIXED_INTERPOLATE, im.qualities.into()),
                })
                .unzip();
            ModelMap::Mixed {
                kinds: Slab::Owned(kinds),
                refs: Slab::Owned(refs),
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ModelMap::Interpolate(v) => v.len(),
            ModelMap::Constant(v) => v.len(),
            ModelMap::Mixed { kinds, .. } => kinds.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the map's records live in a file mapping.
    pub fn is_mapped(&self) -> bool {
        match self {
            ModelMap::Interpolate(v) => v.is_mapped(),
            ModelMap::Constant(v) => v.is_mapped(),
            ModelMap::Mixed { kinds, .. } => kinds.is_mapped(),
        }
    }

    /// Gather the map into a new order: entry `i` of the result is entry
    /// `order[i]` of `self`.  Used to match the BVH's leaf ordering.
    ///
    /// Only an owned map is ever reordered: a mapped one was written in leaf
    /// order already.
    pub(crate) fn reorder(self, order: &[u32]) -> Self {
        fn gather<T: crate::slab::Record>(v: Slab<T>, order: &[u32]) -> Slab<T> {
            Slab::Owned(order.iter().map(|&i| v[i as usize]).collect())
        }
        match self {
            ModelMap::Interpolate(v) => ModelMap::Interpolate(gather(v, order)),
            ModelMap::Constant(v) => ModelMap::Constant(gather(v, order)),
            ModelMap::Mixed { kinds, refs } => ModelMap::Mixed {
                kinds: gather(kinds, order),
                refs: gather(refs, order),
            },
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
            ModelMap::Mixed { kinds, refs } => match kinds[index] {
                MIXED_CONSTANT => qualities[refs[index][0] as usize],
                _ => interpolate_quality(&refs[index], qualities, simplex, point),
            },
        }
    }
}

impl DeepSizeOf for ModelMap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            ModelMap::Interpolate(v) => v.deep_size_of_children(context),
            ModelMap::Constant(v) => v.deep_size_of_children(context),
            ModelMap::Mixed { kinds, refs } => {
                kinds.deep_size_of_children(context) + refs.deep_size_of_children(context)
            }
        }
    }
}
