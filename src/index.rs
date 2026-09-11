//! The on-disk index: a mesh's compiled BVH and records, laid out to be
//! memory-mapped.
//!
//! Building a [`MeshModel`] means reading a mesh off disk and running a
//! surface-area-heuristic BVH build over every simplex.  The result is four
//! flat arrays of fixed-width records, and nothing about them depends on the
//! process that built them, so they can be written once and mapped by every
//! process that needs them afterwards.  Opening an index costs a header read
//! and an `mmap`; the pages a query touches come in on demand and are shared
//! by every process on the node.
//!
//! # Layout
//!
//! ```text
//! page 0        Header, then the model name as UTF-8
//! nodes         [CompactNode]   the tree, in the order the build emitted it
//! simplices     [Simplex]       in BVH leaf order
//! refs          [VertexRefs] or [u32], depending on the model kind
//! kinds         [u32]           only for a Mixed model map
//! qualities     [Quality]       one per mesh vertex
//! ```
//!
//! Every section starts on a [`SECTION_ALIGN`] boundary, which is what lets
//! `zerocopy` cast a mapped range straight to records, and the header records
//! where each one starts.  Numbers are little-endian and `Real` is whatever
//! the extension was built with; the header names the width, and a reader
//! built with the other one refuses the file rather than misread it.
//!
//! The header also holds a fingerprint of the source mesh, supplied by the
//! caller, so that a stale index can be told from a current one without
//! reading the mesh.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

use bvh::aabb::Aabb;
use memmap2::Mmap;
use nalgebra::{Affine3, Matrix4, Point3};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::compact_bvh::{CompactBvh, CompactNode};
use crate::mesh::MeshModel;
use crate::model::{ModelMap, VertexRefs};
use crate::quality::Quality;
use crate::real::Real;
use crate::simplex::Simplex;
use crate::slab::{Record, Slab, SlabError};

/// Identifies the file type.
pub const MAGIC: [u8; 8] = *b"NZCVMIDX";
/// Bumped whenever the layout changes incompatibly.
pub const VERSION: u32 = 1;
/// Every section starts on a boundary this many bytes wide.
///
/// A page, so each section begins page-aligned for the mapping and for any
/// record alignment.  A coarser boundary would suit network filesystems'
/// read sizes; see the design notes.
pub const SECTION_ALIGN: usize = 4096;
/// Length of the source fingerprint the header carries.
pub const FINGERPRINT_LEN: usize = 32;

const KIND_INTERPOLATE: u32 = 0;
const KIND_CONSTANT: u32 = 1;
const KIND_MIXED: u32 = 2;

const FLAG_HAS_ROOT: u32 = 1;
const FLAG_HAS_TRANSFORM: u32 = 2;

// Little-endian is assumed rather than converted; the file records nothing
// about byte order.
const _: () = assert!(cfg!(target_endian = "little"));

/// The first bytes of an index file.
///
/// Fields are ordered widest first so that the `repr(C)` layout has no
/// padding whichever width `Real` has.
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct Header {
    magic: [u8; 8],
    /// Byte offset of each section: nodes, simplices, refs, kinds, qualities.
    offsets: [u64; 5],
    /// Record count of each section, in the same order.
    counts: [u64; 5],
    version: u32,
    /// `size_of::<Real>()` at write time.
    real_width: u32,
    flags: u32,
    model_kind: u32,
    /// The packed root slot; meaningful only with [`FLAG_HAS_ROOT`].
    root_slot: u32,
    priority: u32,
    /// Bytes of UTF-8 name following the header.
    name_len: u32,
    _reserved: u32,
    /// Bounding box as `[min x, y, z, max x, y, z]`.
    aabb: [Real; 6],
    /// Column-major 4x4 affine; meaningful only with [`FLAG_HAS_TRANSFORM`].
    transform: [Real; 16],
    fingerprint: [u8; FINGERPRINT_LEN],
}

// The header and the name have to fit in the first section.
const _: () = assert!(size_of::<Header>() + 256 <= SECTION_ALIGN);

/// Why an index file could not be written or opened.
#[derive(Debug)]
pub enum IndexError {
    Io(io::Error),
    /// The file does not start with [`MAGIC`].
    NotAnIndex,
    /// The file was written by a different layout version.
    Version(u32),
    /// The file was written with a different `Real` width.
    RealWidth(u32),
    /// A section lies outside the file or is misaligned.
    Section(SlabError),
    /// The header is inconsistent with itself.
    Corrupt(&'static str),
    /// The model name is longer than the header page can hold.
    NameTooLong(usize),
}

impl std::fmt::Display for IndexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexError::Io(e) => write!(f, "{e}"),
            IndexError::NotAnIndex => write!(f, "not an NZCVM index file"),
            IndexError::Version(v) => write!(f, "index version {v}, expected {VERSION}"),
            IndexError::RealWidth(w) => write!(
                f,
                "index written with {w}-byte reals, this build uses {}-byte reals",
                size_of::<Real>()
            ),
            IndexError::Section(e) => write!(f, "index section is unreadable: {e}"),
            IndexError::Corrupt(what) => write!(f, "index header is corrupt: {what}"),
            IndexError::NameTooLong(n) => {
                write!(f, "model name of {n} bytes does not fit the header page")
            }
        }
    }
}

impl std::error::Error for IndexError {}

impl From<io::Error> for IndexError {
    fn from(e: io::Error) -> Self {
        IndexError::Io(e)
    }
}

impl From<SlabError> for IndexError {
    fn from(e: SlabError) -> Self {
        IndexError::Section(e)
    }
}

fn align_up(n: usize) -> usize {
    n.div_ceil(SECTION_ALIGN) * SECTION_ALIGN
}

/// Write `records` padded out to the next section boundary.
fn write_section<T: Record, W: Write>(
    out: &mut W,
    written: &mut usize,
    records: &[T],
) -> io::Result<u64> {
    let start = *written;
    let bytes = records.as_bytes();
    out.write_all(bytes)?;
    *written += bytes.len();
    pad_to_boundary(out, written)?;
    Ok(start as u64)
}

fn pad_to_boundary<W: Write>(out: &mut W, written: &mut usize) -> io::Result<()> {
    let target = align_up(*written);
    let pad = vec![0u8; target - *written];
    out.write_all(&pad)?;
    *written = target;
    Ok(())
}

/// Write `model` as an index at `path`.
///
/// The file goes down under a temporary name and moves into place at the end,
/// so a crash mid-write leaves nothing that parses.  `fingerprint` is the
/// caller's summary of the source mesh, stored for [`read_fingerprint`].
pub fn write(
    model: &MeshModel,
    fingerprint: &[u8; FINGERPRINT_LEN],
    path: &Path,
) -> Result<(), IndexError> {
    let name = model.name.as_bytes();
    if size_of::<Header>() + name.len() > SECTION_ALIGN {
        return Err(IndexError::NameTooLong(name.len()));
    }

    let tmp = path.with_extension("nzidx.partial");
    let mut out = io::BufWriter::new(File::create(&tmp)?);

    // Header last: section offsets are only known once they are written, so
    // leave the first page blank and come back to it.
    out.write_all(&vec![0u8; SECTION_ALIGN])?;
    let mut written = SECTION_ALIGN;

    let bvh = model.bvh();
    let nodes_at = write_section(&mut out, &mut written, &bvh.nodes()[..])?;
    let simplices_at = write_section(&mut out, &mut written, &model.simplices()[..])?;
    let (model_kind, refs_at, refs_len, kinds_at, kinds_len) = match model.model_map() {
        ModelMap::Interpolate(refs) => {
            let at = write_section(&mut out, &mut written, &refs[..])?;
            (KIND_INTERPOLATE, at, refs.len(), 0, 0)
        }
        ModelMap::Constant(refs) => {
            let at = write_section(&mut out, &mut written, &refs[..])?;
            (KIND_CONSTANT, at, refs.len(), 0, 0)
        }
        ModelMap::Mixed { kinds, refs } => {
            let refs_at = write_section(&mut out, &mut written, &refs[..])?;
            let kinds_at = write_section(&mut out, &mut written, &kinds[..])?;
            (KIND_MIXED, refs_at, refs.len(), kinds_at, kinds.len())
        }
    };
    let qualities_at = write_section(&mut out, &mut written, &model.qualities()[..])?;

    let aabb = model.aabb3();
    let mut flags = 0;
    let mut transform = [0 as Real; 16];
    if let Some(affine) = model.transform() {
        flags |= FLAG_HAS_TRANSFORM;
        transform.copy_from_slice(affine.matrix().as_slice());
    }
    let root_slot = match bvh.root_bits() {
        Some(bits) => {
            flags |= FLAG_HAS_ROOT;
            bits
        }
        None => 0,
    };

    let header = Header {
        magic: MAGIC,
        offsets: [nodes_at, simplices_at, refs_at, kinds_at, qualities_at],
        counts: [
            bvh.nodes().len() as u64,
            model.simplices().len() as u64,
            refs_len as u64,
            kinds_len as u64,
            model.qualities().len() as u64,
        ],
        version: VERSION,
        real_width: size_of::<Real>() as u32,
        flags,
        model_kind,
        root_slot,
        priority: model.priority as u32,
        name_len: name.len() as u32,
        _reserved: 0,
        aabb: [
            aabb.min.x, aabb.min.y, aabb.min.z, aabb.max.x, aabb.max.y, aabb.max.z,
        ],
        transform,
        fingerprint: *fingerprint,
    };

    let mut file = out.into_inner().map_err(|e| e.into_error())?;
    use std::io::Seek;
    file.seek(io::SeekFrom::Start(0))?;
    file.write_all(header.as_bytes())?;
    file.write_all(name)?;
    file.sync_all()?;
    drop(file);

    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Read and check the header of the file at `path`, and the name after it.
fn read_header(path: &Path) -> Result<(Header, String), IndexError> {
    let mut file = File::open(path)?;
    let mut page = vec![0u8; SECTION_ALIGN];
    let mut filled = 0;
    while filled < page.len() {
        match file.read(&mut page[filled..])? {
            0 => break,
            n => filled += n,
        }
    }
    parse_header(&page[..filled])
}

fn parse_header(page: &[u8]) -> Result<(Header, String), IndexError> {
    let (header, rest) = Header::ref_from_prefix(page).map_err(|_| IndexError::NotAnIndex)?;
    if header.magic != MAGIC {
        return Err(IndexError::NotAnIndex);
    }
    if header.version != VERSION {
        return Err(IndexError::Version(header.version));
    }
    if header.real_width as usize != size_of::<Real>() {
        return Err(IndexError::RealWidth(header.real_width));
    }
    let name_len = header.name_len as usize;
    let name_bytes = rest
        .get(..name_len)
        .ok_or(IndexError::Corrupt("name runs past the header page"))?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| IndexError::Corrupt("name is not UTF-8"))?
        .to_owned();
    Ok((*header, name))
}

/// The source fingerprint stored in the index at `path`.
///
/// Reads only the header page, so it is cheap enough to run before deciding
/// whether to open the index or rebuild.
pub fn read_fingerprint(path: &Path) -> Result<[u8; FINGERPRINT_LEN], IndexError> {
    Ok(read_header(path)?.0.fingerprint)
}

fn section<T: Record>(mmap: &Arc<Mmap>, header: &Header, i: usize) -> Result<Slab<T>, IndexError> {
    let offset = usize::try_from(header.offsets[i])
        .map_err(|_| IndexError::Corrupt("section offset overflows"))?;
    let len = usize::try_from(header.counts[i])
        .map_err(|_| IndexError::Corrupt("section count overflows"))?;
    Ok(Slab::mapped(Arc::clone(mmap), offset, len)?)
}

/// Open the index at `path` as a memory-mapped [`MeshModel`].
pub fn open(path: &Path) -> Result<MeshModel, IndexError> {
    let (header, name) = read_header(path)?;
    let file = File::open(path)?;
    // SAFETY: an index file is written once and never modified in place
    // (`write` renames a complete temporary into position), so the mapping
    // cannot observe a change underneath it.  Truncating or replacing the
    // file while it is mapped is the documented hazard of this format.
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });

    let nodes: Slab<CompactNode> = section(&mmap, &header, 0)?;
    let simplices: Slab<Simplex> = section(&mmap, &header, 1)?;
    let qualities: Slab<Quality> = section(&mmap, &header, 4)?;
    let model_map = match header.model_kind {
        KIND_INTERPOLATE => ModelMap::Interpolate(section::<VertexRefs>(&mmap, &header, 2)?),
        KIND_CONSTANT => ModelMap::Constant(section::<u32>(&mmap, &header, 2)?),
        KIND_MIXED => ModelMap::Mixed {
            refs: section::<VertexRefs>(&mmap, &header, 2)?,
            kinds: section::<u32>(&mmap, &header, 3)?,
        },
        _ => return Err(IndexError::Corrupt("unknown model kind")),
    };
    if model_map.len() != simplices.len() {
        return Err(IndexError::Corrupt(
            "model map and simplices differ in length",
        ));
    }

    let root = (header.flags & FLAG_HAS_ROOT != 0).then_some(header.root_slot);
    let bvh_tree = CompactBvh::from_parts(nodes, root);
    let [x0, y0, z0, x1, y1, z1] = header.aabb;
    let aabb = Aabb::with_bounds(Point3::new(x0, y0, z0), Point3::new(x1, y1, z1));
    let transform = (header.flags & FLAG_HAS_TRANSFORM != 0)
        .then(|| Affine3::from_matrix_unchecked(Matrix4::from_column_slice(&header.transform)));
    let priority = u8::try_from(header.priority)
        .map_err(|_| IndexError::Corrupt("priority does not fit a byte"))?;

    Ok(MeshModel::from_parts(
        bvh_tree, simplices, model_map, qualities, aabb, transform, priority, name,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ConstantModel, InterpolateModel, Model};
    use nalgebra::Point4;
    use proptest::prelude::*;

    /// A 3x3x3 curvilinear mesh over the unit cube with a linear quality field.
    fn cube_mesh() -> MeshModel {
        let n = 3usize;
        let mut vertices = Vec::new();
        let mut qualities = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let (x, y, z) = (
                        i as Real / (n - 1) as Real,
                        j as Real / (n - 1) as Real,
                        k as Real / (n - 1) as Real,
                    );
                    vertices.push(Point3::new(x, y, z));
                    qualities.push(Quality {
                        rho: 1000.0 + x,
                        vp: 2000.0 + y,
                        vs: 500.0 + z,
                        qp: 100.0,
                        qs: 50.0,
                        alpha: 1.0,
                    });
                }
            }
        }
        MeshModel::curvilinear_mesh(vertices, qualities, (n, n, n), |i, j, k| {
            i * n * n + j * n + k
        })
        .ok()
        .expect("cube mesh")
    }

    fn fingerprint() -> [u8; FINGERPRINT_LEN] {
        let mut f = [0u8; FINGERPRINT_LEN];
        for (i, b) in f.iter_mut().enumerate() {
            *b = i as u8;
        }
        f
    }

    fn round_trip(model: &MeshModel) -> (tempfile::TempDir, MeshModel) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.nzidx");
        write(model, &fingerprint(), &path).unwrap();
        let opened = open(&path).unwrap();
        assert!(opened.is_mapped());
        (dir, opened)
    }

    fn same_quality(a: Option<Quality>, b: Option<Quality>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }

    #[test]
    fn a_mapped_model_answers_like_the_built_one() {
        let built = cube_mesh();
        let (_dir, mapped) = round_trip(&built);
        for &(x, y, z) in &[
            (0.1, 0.2, 0.3),
            (0.5, 0.5, 0.5),
            (0.9, 0.1, 0.7),
            (1.5, 0.5, 0.5),
        ] {
            let p = Point3::new(x, y, z);
            assert!(same_quality(built.query(p), mapped.query(p)), "at {p}");
        }
    }

    #[test]
    fn the_header_round_trips_the_metadata() {
        let built = cube_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.nzidx");
        write(&built, &fingerprint(), &path).unwrap();
        let opened = open(&path).unwrap();
        assert_eq!(opened.name, built.name);
        assert_eq!(opened.priority, built.priority);
        assert_eq!(read_fingerprint(&path).unwrap(), fingerprint());
        let (a, b) = (built.aabb3(), opened.aabb3());
        assert_eq!(a.min, b.min);
        assert_eq!(a.max, b.max);
    }

    #[test]
    fn a_mixed_model_map_round_trips() {
        // Two tetrahedra sharing a face, one constant and one interpolating.
        let vertices = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
            Point3::new(1.0, 1.0, 1.0),
        ];
        let faces = vec![Point4::new(0usize, 1, 2, 3), Point4::new(1usize, 2, 3, 4)];
        let q = |v: Real| Quality {
            rho: v,
            vp: v,
            vs: v,
            qp: 1.0,
            qs: 1.0,
            alpha: 1.0,
        };
        let qualities = vec![q(1.0), q(2.0), q(3.0), q(4.0), q(5.0), q(99.0)];
        let models = vec![
            Model::Constant(ConstantModel { quality: 5 }),
            Model::Interpolate(InterpolateModel {
                qualities: Point4::new(1, 2, 3, 4),
            }),
        ];
        let built = MeshModel::new(vertices, faces, models, qualities, 7, None, "mixed".into())
            .ok()
            .expect("non-degenerate");
        assert!(matches!(built.model_map(), ModelMap::Mixed { .. }));
        let (_dir, mapped) = round_trip(&built);
        assert!(matches!(mapped.model_map(), ModelMap::Mixed { .. }));
        for &(x, y, z) in &[(0.1, 0.1, 0.1), (0.6, 0.6, 0.6)] {
            let p = Point3::new(x, y, z);
            assert!(same_quality(built.query(p), mapped.query(p)), "at {p}");
        }
    }

    #[test]
    fn an_empty_model_round_trips() {
        let built = MeshModel::new(vec![], vec![], vec![], vec![], 1, None, "empty".into())
            .ok()
            .expect("empty mesh builds");
        let (_dir, mapped) = round_trip(&built);
        assert!(mapped.query(Point3::new(0.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn a_foreign_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.nzidx");
        std::fs::write(&path, vec![0u8; SECTION_ALIGN]).unwrap();
        assert!(matches!(open(&path), Err(IndexError::NotAnIndex)));
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let built = cube_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.nzidx");
        write(&built, &fingerprint(), &path).unwrap();
        // The last section is padded to a boundary, so trimming a little off
        // the end still leaves every record in place. Cut into the records.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        assert!(matches!(open(&path), Err(IndexError::Section(_))));
    }

    #[test]
    fn another_real_width_is_refused() {
        let built = cube_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.nzidx");
        write(&built, &fingerprint(), &path).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let (header, _) = Header::mut_from_prefix(&mut bytes).unwrap();
        header.real_width = if size_of::<Real>() == 4 { 8 } else { 4 };
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(open(&path), Err(IndexError::RealWidth(_))));
    }

    #[test]
    fn sections_start_on_boundaries() {
        let built = cube_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mesh.nzidx");
        write(&built, &fingerprint(), &path).unwrap();
        let (header, _) = read_header(&path).unwrap();
        for (i, off) in header.offsets.iter().enumerate() {
            if header.counts[i] > 0 {
                assert_eq!(*off as usize % SECTION_ALIGN, 0, "section {i}");
            }
        }
    }

    proptest! {
        /// Built and mapped agree everywhere in and around the cube.
        #[test]
        fn prop_mapped_agrees_with_built(
            x in (-0.5 as Real)..1.5,
            y in (-0.5 as Real)..1.5,
            z in (-0.5 as Real)..1.5,
        ) {
            let built = cube_mesh();
            let (_dir, mapped) = round_trip(&built);
            let p = Point3::new(x, y, z);
            prop_assert!(same_quality(built.query(p), mapped.query(p)), "at {p}");
        }
    }
}
