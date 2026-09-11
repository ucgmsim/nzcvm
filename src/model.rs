use crate::index::{IndexError, SectionReader, SectionWriter, tag};
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

/// Per-mesh map from simplex index to its model.
///
/// Meshes are almost always homogeneous (basins are all-interpolate), so the
/// common cases store just the quality indices in a flat array — 16 bytes per
/// simplex for interpolation, 4 for constant — instead of a `Vec<Model>` that
/// pays an enum tag per element.  A heterogeneous mesh stores four indices
/// per simplex too, and marks a constant one with [`NO_INTERPOLATION`] in the
/// second slot.
///
/// Every variant is a [`Slab`] so that the map can be memory-mapped from an
/// index file as readily as built in memory.
pub enum ModelMap {
    Interpolate(Slab<VertexRefs>),
    Constant(Slab<u32>),
    Mixed(Slab<VertexRefs>),
}

/// Marks a constant simplex in a [`ModelMap::Mixed`] map.
///
/// Sits in the second index slot, where an interpolating simplex has a
/// vertex index.  A mesh holds at most `u32::MAX` qualities, so no vertex
/// index is ever this value.
pub const NO_INTERPOLATION: u32 = u32::MAX;

const KIND_INTERPOLATE: u32 = 0;
const KIND_CONSTANT: u32 = 1;
const KIND_MIXED: u32 = 2;

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
            ModelMap::Mixed(Slab::Owned(
                models
                    .into_iter()
                    .map(|m| match m {
                        Model::Constant(cm) => [cm.quality, NO_INTERPOLATION, 0, 0],
                        Model::Interpolate(im) => im.qualities.into(),
                    })
                    .collect(),
            ))
        }
    }

    pub fn len(&self) -> usize {
        match self {
            ModelMap::Interpolate(v) | ModelMap::Mixed(v) => v.len(),
            ModelMap::Constant(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
            ModelMap::Mixed(v) => ModelMap::Mixed(gather(v, order)),
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
            ModelMap::Mixed(v) => {
                let refs = &v[index];
                if refs[1] == NO_INTERPOLATION {
                    qualities[refs[0] as usize]
                } else {
                    interpolate_quality(refs, qualities, simplex, point)
                }
            }
        }
    }

    /// The kind tag the index header records for this map.
    pub(crate) fn kind(&self) -> u32 {
        match self {
            ModelMap::Interpolate(_) => KIND_INTERPOLATE,
            ModelMap::Constant(_) => KIND_CONSTANT,
            ModelMap::Mixed(_) => KIND_MIXED,
        }
    }

    /// Append this map's records to an index being written.
    pub(crate) fn write_sections(&self, writer: &mut SectionWriter) -> Result<(), IndexError> {
        match self {
            ModelMap::Interpolate(v) | ModelMap::Mixed(v) => writer.write(tag::REFS, v),
            ModelMap::Constant(v) => writer.write(tag::REFS, v),
        }
    }

    /// Map this map's records back out of an opened index.
    pub(crate) fn read_sections(reader: &SectionReader) -> Result<Self, IndexError> {
        Ok(match reader.model_kind() {
            KIND_INTERPOLATE => ModelMap::Interpolate(reader.section(tag::REFS)?),
            KIND_CONSTANT => ModelMap::Constant(reader.section(tag::REFS)?),
            KIND_MIXED => ModelMap::Mixed(reader.section(tag::REFS)?),
            _ => return Err(IndexError::Corrupt("unknown model kind")),
        })
    }
}

impl DeepSizeOf for ModelMap {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        match self {
            ModelMap::Interpolate(v) | ModelMap::Mixed(v) => v.deep_size_of_children(context),
            ModelMap::Constant(v) => v.deep_size_of_children(context),
        }
    }
}
