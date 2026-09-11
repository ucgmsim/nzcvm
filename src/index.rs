//! The on-disk index file: a header page and a table of record sections.
//!
//! Building a [`MeshModel`](crate::mesh::MeshModel) means reading a mesh off
//! disk and running a surface-area-heuristic BVH build over every simplex.
//! The result is a handful of flat arrays of fixed-width records, and nothing
//! about them depends on the process that built them, so they can be written
//! once and mapped by every process that needs them afterwards.  Opening an
//! index costs a header read and an `mmap`; the pages a query touches come in
//! on demand and are shared by every process on the node.
//!
//! This module owns the file mechanics and nothing about meshes.  A
//! [`SectionWriter`] appends record arrays and notes where each landed; a
//! [`SectionReader`] maps the file and hands back a
//! [`Slab`](crate::slab::Slab) per section.  The types that own the records
//! serialise themselves through those two, so their fields stay private.
//!
//! # Layout
//!
//! ```text
//! page 0        Header, then the model name as UTF-8
//! sections...   one record array each, in the order they were written
//! ```
//!
//! Every section starts on a [`SECTION_ALIGN`] boundary, which is what lets
//! `zerocopy` cast a mapped range straight to records.  The header holds one
//! [`Section`] descriptor per array: where it starts, how many records, how
//! wide each record is, and a tag saying which array it is.  A reader looks a
//! section up by tag and refuses one whose stride differs from its own record
//! type, so a layout skew is caught at open rather than misread.
//!
//! Numbers are little-endian, and `Real` is whatever the extension was built
//! with; the header names the width, and a reader built with the other one
//! refuses the file.  The header also holds a fingerprint of the source mesh,
//! supplied by the caller, so a stale index can be told from a current one
//! without reading the mesh.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bvh::aabb::Aabb;
use memmap2::Mmap;
use nalgebra::{Affine3, Matrix4, Point3};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

use crate::real::Real;
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
/// Room in the header's section table.
pub const MAX_SECTIONS: usize = 8;

/// Tags naming what a section holds.
pub mod tag {
    pub const NODES: u32 = 1;
    pub const SIMPLICES: u32 = 2;
    pub const REFS: u32 = 3;
    pub const QUALITIES: u32 = 4;
}

const FLAG_HAS_ROOT: u32 = 1;
const FLAG_HAS_TRANSFORM: u32 = 2;

// Little-endian is assumed rather than converted; the file records nothing
// about byte order.
const _: () = assert!(cfg!(target_endian = "little"));

/// Where one record array sits in the file.
#[derive(Clone, Copy, Debug, Default, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct Section {
    offset: u64,
    count: u64,
    /// `size_of` the record type at write time.
    stride: u32,
    /// One of the [`tag`] constants, or zero for an unused table entry.
    tag: u32,
}

/// The first bytes of an index file.
///
/// Fields are ordered widest first so that the `repr(C)` layout has no
/// padding whichever width `Real` has.
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
struct Header {
    magic: [u8; 8],
    sections: [Section; MAX_SECTIONS],
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
    section_count: u32,
    aabb_min: [Real; 3],
    aabb_max: [Real; 3],
    /// Column-major 4x4 affine; meaningful only with [`FLAG_HAS_TRANSFORM`].
    transform: [Real; 16],
    fingerprint: [u8; FINGERPRINT_LEN],
}

// The header and a reasonable name have to fit in the first section.
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
    /// The header is inconsistent with itself or with the reader.
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

/// The per-model fields of a header, supplied by the type being written.
pub struct IndexMeta<'a> {
    pub model_kind: u32,
    pub root_slot: Option<u32>,
    pub priority: u8,
    pub name: &'a str,
    pub aabb: Aabb<Real, 3>,
    pub transform: Option<Affine3<Real>>,
    pub fingerprint: &'a [u8; FINGERPRINT_LEN],
}

/// Appends record sections to a new index file.
///
/// The file goes down under a `.partial` name and [`SectionWriter::finish`]
/// renames it into place, so a crash mid-write leaves nothing that parses.
pub struct SectionWriter {
    out: BufWriter<File>,
    written: usize,
    sections: Vec<Section>,
    path: PathBuf,
    partial: PathBuf,
}

impl SectionWriter {
    /// Start writing the index that will end up at `path`.
    pub fn create(path: &Path) -> Result<Self, IndexError> {
        let partial = path.with_extension("nzidx.partial");
        let mut out = BufWriter::new(File::create(&partial)?);
        // The header is written last, once every section offset is known, so
        // the first page is left blank for now.
        pad(&mut out, SECTION_ALIGN)?;
        Ok(Self {
            out,
            written: SECTION_ALIGN,
            sections: Vec::new(),
            path: path.to_owned(),
            partial,
        })
    }

    /// Append `records` as the section tagged `tag`.
    pub fn write<T: Record>(&mut self, tag: u32, records: &[T]) -> Result<(), IndexError> {
        if self.sections.len() == MAX_SECTIONS {
            return Err(IndexError::Corrupt(
                "too many sections for the header table",
            ));
        }
        let bytes = records.as_bytes();
        self.sections.push(Section {
            offset: self.written as u64,
            count: records.len() as u64,
            stride: size_of::<T>() as u32,
            tag,
        });
        self.out.write_all(bytes)?;
        self.written += bytes.len();
        let target = self.written.next_multiple_of(SECTION_ALIGN);
        pad(&mut self.out, target - self.written)?;
        self.written = target;
        Ok(())
    }

    /// Write the header and move the file into place.
    pub fn finish(self, meta: &IndexMeta<'_>) -> Result<(), IndexError> {
        let name = meta.name.as_bytes();
        if size_of::<Header>() + name.len() > SECTION_ALIGN {
            return Err(IndexError::NameTooLong(name.len()));
        }

        let mut flags = 0;
        let mut transform = [0 as Real; 16];
        if let Some(affine) = meta.transform {
            flags |= FLAG_HAS_TRANSFORM;
            transform.copy_from_slice(affine.matrix().as_slice());
        }
        if meta.root_slot.is_some() {
            flags |= FLAG_HAS_ROOT;
        }
        let mut sections = [Section::default(); MAX_SECTIONS];
        sections[..self.sections.len()].copy_from_slice(&self.sections);

        let header = Header {
            magic: MAGIC,
            sections,
            version: VERSION,
            real_width: size_of::<Real>() as u32,
            flags,
            model_kind: meta.model_kind,
            root_slot: meta.root_slot.unwrap_or(0),
            priority: meta.priority as u32,
            name_len: name.len() as u32,
            section_count: self.sections.len() as u32,
            aabb_min: meta.aabb.min.into(),
            aabb_max: meta.aabb.max.into(),
            transform,
            fingerprint: *meta.fingerprint,
        };

        let mut file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.seek(io::SeekFrom::Start(0))?;
        file.write_all(header.as_bytes())?;
        file.write_all(name)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&self.partial, &self.path)?;
        Ok(())
    }
}

fn pad<W: Write>(out: &mut W, bytes: usize) -> io::Result<()> {
    io::copy(&mut io::repeat(0).take(bytes as u64), out)?;
    Ok(())
}

/// Read and check the first page of the file at `path`.
fn read_header(path: &Path) -> Result<(Header, String), IndexError> {
    let mut page = Vec::with_capacity(SECTION_ALIGN);
    File::open(path)?
        .take(SECTION_ALIGN as u64)
        .read_to_end(&mut page)?;
    let (header, rest) = Header::ref_from_prefix(&page).map_err(|_| IndexError::NotAnIndex)?;
    if header.magic != MAGIC {
        return Err(IndexError::NotAnIndex);
    }
    if header.version != VERSION {
        return Err(IndexError::Version(header.version));
    }
    if header.real_width as usize != size_of::<Real>() {
        return Err(IndexError::RealWidth(header.real_width));
    }
    if header.section_count as usize > MAX_SECTIONS {
        return Err(IndexError::Corrupt("section count exceeds the table"));
    }
    let name = rest
        .get(..header.name_len as usize)
        .ok_or(IndexError::Corrupt("name runs past the header page"))
        .and_then(|b| std::str::from_utf8(b).map_err(|_| IndexError::Corrupt("name is not UTF-8")))?
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

/// A mapped index file, handing out its sections as slabs.
pub struct SectionReader {
    mmap: Arc<Mmap>,
    header: Header,
    name: String,
}

impl SectionReader {
    /// Check the header of the file at `path` and map the rest.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        let (header, name) = read_header(path)?;
        let file = File::open(path)?;
        // SAFETY: an index file is written once and never modified in place
        // (`SectionWriter::finish` renames a complete temporary into
        // position), so the mapping cannot observe a change underneath it.
        // Truncating or replacing the file while it is mapped is the
        // documented hazard of this format.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        Ok(Self { mmap, header, name })
    }

    /// The section tagged `tag`, as records of type `T`.
    ///
    /// Refuses a section whose stride is not `size_of::<T>()`: the file was
    /// written with a different record layout than this reader expects.
    pub fn section<T: Record>(&self, tag: u32) -> Result<Slab<T>, IndexError> {
        let section = self.header.sections[..self.header.section_count as usize]
            .iter()
            .find(|s| s.tag == tag)
            .ok_or(IndexError::Corrupt("missing section"))?;
        if section.stride as usize != size_of::<T>() {
            return Err(IndexError::Corrupt(
                "section stride does not match its record type",
            ));
        }
        let offset = usize::try_from(section.offset)
            .map_err(|_| IndexError::Corrupt("section offset overflows"))?;
        let len = usize::try_from(section.count)
            .map_err(|_| IndexError::Corrupt("section count overflows"))?;
        Ok(Slab::mapped(Arc::clone(&self.mmap), offset, len)?)
    }

    pub fn model_kind(&self) -> u32 {
        self.header.model_kind
    }

    pub fn root_slot(&self) -> Option<u32> {
        (self.header.flags & FLAG_HAS_ROOT != 0).then_some(self.header.root_slot)
    }

    pub fn priority(&self) -> Result<u8, IndexError> {
        u8::try_from(self.header.priority)
            .map_err(|_| IndexError::Corrupt("priority does not fit a byte"))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn aabb(&self) -> Aabb<Real, 3> {
        Aabb::with_bounds(
            Point3::from(self.header.aabb_min),
            Point3::from(self.header.aabb_max),
        )
    }

    pub fn transform(&self) -> Option<Affine3<Real>> {
        (self.header.flags & FLAG_HAS_TRANSFORM != 0).then(|| {
            Affine3::from_matrix_unchecked(Matrix4::from_column_slice(&self.header.transform))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshModel;
    use crate::mesh::tests::{generate_grid, mock_quality};
    use crate::model::{ConstantModel, InterpolateModel, Model, ModelMap};
    use crate::quality::Quality;
    use nalgebra::Point4;
    use proptest::prelude::*;

    /// A 3x3x3 curvilinear mesh with a quality field that varies by vertex.
    fn cube_mesh() -> MeshModel {
        let n = 3usize;
        let vertices = generate_grid(n, n, n);
        let qualities = vertices
            .iter()
            .map(|p| mock_quality(p.x + 2.0 * p.y + 3.0 * p.z))
            .collect();
        MeshModel::curvilinear_mesh(vertices, qualities, (n, n, n), |i, j, k| {
            k * n * n + j * n + i
        })
        .ok()
        .expect("cube mesh")
    }

    fn fingerprint() -> [u8; FINGERPRINT_LEN] {
        std::array::from_fn(|i| i as u8)
    }

    fn write_to(model: &MeshModel, dir: &tempfile::TempDir) -> PathBuf {
        let path = dir.path().join("mesh.nzidx");
        model.write_index(&fingerprint(), &path).unwrap();
        path
    }

    fn round_trip(model: &MeshModel) -> (tempfile::TempDir, MeshModel) {
        let dir = tempfile::tempdir().unwrap();
        let opened = MeshModel::open_index(&write_to(model, &dir)).unwrap();
        assert!(opened.is_mapped());
        (dir, opened)
    }

    #[test]
    fn a_mapped_model_answers_like_the_built_one() {
        let built = cube_mesh();
        let (_dir, mapped) = round_trip(&built);
        for &(x, y, z) in &[
            (0.1, 0.2, 0.3),
            (1.0, 1.0, 1.0),
            (1.9, 0.1, 1.7),
            (2.5, 0.5, 0.5),
        ] {
            let p = Point3::new(x, y, z);
            assert_eq!(built.query(p), mapped.query(p), "at {p}");
        }
    }

    #[test]
    fn the_header_round_trips_the_metadata() {
        let built = cube_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = write_to(&built, &dir);
        let opened = MeshModel::open_index(&path).unwrap();
        assert_eq!(opened.name, built.name);
        assert_eq!(opened.priority, built.priority);
        assert_eq!(read_fingerprint(&path).unwrap(), fingerprint());
        assert_eq!(opened.aabb3().min, built.aabb3().min);
        assert_eq!(opened.aabb3().max, built.aabb3().max);
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
        let qualities: Vec<Quality> = [1.0, 2.0, 3.0, 4.0, 5.0, 99.0]
            .into_iter()
            .map(mock_quality)
            .collect();
        let models = vec![
            Model::from(ConstantModel { quality: 5 }),
            Model::from(InterpolateModel {
                qualities: Point4::new(1, 2, 3, 4),
            }),
        ];
        let built = MeshModel::new(vertices, faces, models, qualities, 7, None, "mixed".into())
            .ok()
            .expect("non-degenerate");
        assert!(matches!(built.model_map(), ModelMap::Mixed(_)));
        let (_dir, mapped) = round_trip(&built);
        assert!(matches!(mapped.model_map(), ModelMap::Mixed(_)));
        for &(x, y, z) in &[(0.1, 0.1, 0.1), (0.6, 0.6, 0.6)] {
            let p = Point3::new(x, y, z);
            assert_eq!(built.query(p), mapped.query(p), "at {p}");
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
        assert!(matches!(
            MeshModel::open_index(&path),
            Err(IndexError::NotAnIndex)
        ));
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_to(&cube_mesh(), &dir);
        // The last section is padded to a boundary, so trimming a little off
        // the end still leaves every record in place. Cut into the records.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();
        assert!(matches!(
            MeshModel::open_index(&path),
            Err(IndexError::Section(_))
        ));
    }

    #[test]
    fn another_real_width_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_to(&cube_mesh(), &dir);
        let mut bytes = std::fs::read(&path).unwrap();
        let (header, _) = Header::mut_from_prefix(&mut bytes).unwrap();
        header.real_width = if size_of::<Real>() == 4 { 8 } else { 4 };
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            MeshModel::open_index(&path),
            Err(IndexError::RealWidth(_))
        ));
    }

    #[test]
    fn a_section_of_the_wrong_stride_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_to(&cube_mesh(), &dir);
        let mut bytes = std::fs::read(&path).unwrap();
        let (header, _) = Header::mut_from_prefix(&mut bytes).unwrap();
        header.sections[0].stride += 4;
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            MeshModel::open_index(&path),
            Err(IndexError::Corrupt(_))
        ));
    }

    #[test]
    fn sections_start_on_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_to(&cube_mesh(), &dir);
        let (header, _) = read_header(&path).unwrap();
        for section in &header.sections[..header.section_count as usize] {
            assert_eq!(
                section.offset as usize % SECTION_ALIGN,
                0,
                "tag {}",
                section.tag
            );
        }
    }

    proptest! {
        /// Built and mapped agree everywhere in and around the cube.
        #[test]
        fn prop_mapped_agrees_with_built(
            x in (-0.5 as Real)..2.5,
            y in (-0.5 as Real)..2.5,
            z in (-0.5 as Real)..2.5,
        ) {
            let built = cube_mesh();
            let (_dir, mapped) = round_trip(&built);
            let p = Point3::new(x, y, z);
            prop_assert_eq!(built.query(p), mapped.query(p), "at {}", p);
        }
    }
}
