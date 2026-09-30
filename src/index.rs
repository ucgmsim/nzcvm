//! The on-disk index file: a header page and a table of record sections.
//!
//! Building a [`MeshModel`](crate::mesh::MeshModel) means reading a mesh off
//! disk and running a surface-area-heuristic BVH build over every simplex.
//! The result is a handful of flat arrays of fixed-width records, and nothing
//! about them depends on the process that built them, so they can be written
//! once and mapped by every process that needs them afterwards.  Opening an
//! index costs one `open` and an `mmap`; the pages a query touches come in on
//! demand and are shared by every process on the node.
//!
//! This module owns the file mechanics and nothing about what the records
//! mean.  A [`SectionWriter`] appends record arrays under caller-chosen tags;
//! a [`SectionReader`] maps the file and hands back a
//! [`Slab`](crate::slab::Slab) per tag.
//!
//! # Layout
//!
//! ```text
//! page 0        Header
//! sections...   one record array each, in the order they were written
//! ```
//!
//! Every section starts on a [`SECTION_ALIGN`] boundary, which is what lets
//! `bytemuck` cast a mapped range straight to records.  The header holds one
//! [`Section`] descriptor per array: where it starts, how many records, how
//! wide each record is, and the caller's tag for it.  A reader looks a
//! section up by tag and refuses one whose stride differs from its own record
//! type, so a layout skew is caught at open rather than misread.
//!
//! Numbers are little-endian, and `Real` is whatever the extension was built
//! with; the header names the width, and a reader built with the other one
//! refuses the file.  The header also holds a fingerprint of the source,
//! supplied by the caller, and [`SectionReader::open`] refuses an index whose
//! fingerprint differs from the one expected, so a stale index is never used.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;

use crate::real::Real;
use crate::slab::{Record, Slab};

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

/// A source fingerprint.
pub type Fingerprint = [u8; FINGERPRINT_LEN];

// Little-endian is assumed rather than converted; the file records nothing
// about byte order.
const _: () = assert!(cfg!(target_endian = "little"));

/// Where one record array sits in the file.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
struct Section {
    offset: u64,
    count: u64,
    /// `size_of` the record type at write time.
    stride: u32,
    /// The caller's tag, or zero for an unused table entry.
    tag: u32,
}

/// The first bytes of an index file.
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct Header {
    magic: [u8; 8],
    sections: [Section; MAX_SECTIONS],
    version: u32,
    /// `size_of::<Real>()` at write time.
    real_width: u32,
    fingerprint: Fingerprint,
}

const _: () = assert!(size_of::<Header>() <= SECTION_ALIGN);

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
    /// The file was built from a different source than the one expected.
    Stale,
    /// The file is inconsistent with itself or with the reader.
    Corrupt(&'static str),
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
            IndexError::Stale => write!(f, "index was built from a different source"),
            IndexError::Corrupt(what) => write!(f, "index is corrupt: {what}"),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<io::Error> for IndexError {
    fn from(e: io::Error) -> Self {
        IndexError::Io(e)
    }
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
    ///
    /// # Panics
    ///
    /// If `tag` is zero or more than [`MAX_SECTIONS`] sections are written:
    /// both are fixed by the caller's code, not by the data.
    pub fn write<T: Record>(&mut self, tag: u32, records: &[T]) -> Result<(), IndexError> {
        assert_ne!(tag, 0, "tag zero marks an unused section");
        assert!(self.sections.len() < MAX_SECTIONS, "too many sections");
        let bytes: &[u8] = bytemuck::cast_slice(records);
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
    pub fn finish(self, fingerprint: &Fingerprint) -> Result<(), IndexError> {
        let mut sections = [Section::default(); MAX_SECTIONS];
        sections[..self.sections.len()].copy_from_slice(&self.sections);
        let header = Header {
            magic: MAGIC,
            sections,
            version: VERSION,
            real_width: size_of::<Real>() as u32,
            fingerprint: *fingerprint,
        };

        let mut file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.seek(io::SeekFrom::Start(0))?;
        file.write_all(bytemuck::bytes_of(&header))?;
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

/// A mapped index file, handing out its sections as slabs.
pub struct SectionReader {
    mmap: Arc<Mmap>,
    header: Header,
}

impl SectionReader {
    /// Map the file at `path` and check its header, including that it was
    /// built from the source `fingerprint` describes.
    pub fn open(path: &Path, fingerprint: &Fingerprint) -> Result<Self, IndexError> {
        let file = File::open(path)?;
        // SAFETY: an index file is written once and never modified in place
        // (`SectionWriter::finish` renames a complete temporary into
        // position), so the mapping cannot observe a change underneath it.
        // Truncating or replacing the file while it is mapped is the
        // documented hazard of this format.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let header: Header = mmap
            .get(..size_of::<Header>())
            .map(bytemuck::pod_read_unaligned)
            .ok_or(IndexError::NotAnIndex)?;
        if header.magic != MAGIC {
            return Err(IndexError::NotAnIndex);
        }
        if header.version != VERSION {
            return Err(IndexError::Version(header.version));
        }
        if header.real_width as usize != size_of::<Real>() {
            return Err(IndexError::RealWidth(header.real_width));
        }
        if header.fingerprint != *fingerprint {
            return Err(IndexError::Stale);
        }
        Ok(Self { mmap, header })
    }

    fn find(&self, tag: u32) -> Option<&Section> {
        self.header
            .sections
            .iter()
            .find(|s| s.tag == tag && tag != 0)
    }

    /// Whether the file has a section tagged `tag`.
    pub fn contains(&self, tag: u32) -> bool {
        self.find(tag).is_some()
    }

    /// The section tagged `tag`, as records of type `T`.
    ///
    /// Refuses a section whose stride is not `size_of::<T>()`: the file was
    /// written with a different record layout than this reader expects.
    pub fn section<T: Record>(&self, tag: u32) -> Result<Slab<T>, IndexError> {
        let section = self
            .find(tag)
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
        Slab::mapped(Arc::clone(&self.mmap), offset, len).ok_or(IndexError::Corrupt(
            "section runs past the end of the file or is misaligned",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u32 = 1;
    const B: u32 = 2;
    const FINGERPRINT: Fingerprint = [7; FINGERPRINT_LEN];

    /// Write a small file with a `u32` section and a `u64` section.
    fn write_sample(dir: &tempfile::TempDir) -> PathBuf {
        let path = dir.path().join("sample.nzidx");
        let mut writer = SectionWriter::create(&path).unwrap();
        writer.write(A, &(0..1000u32).collect::<Vec<_>>()).unwrap();
        writer.write(B, &[1u64, 2, 3]).unwrap();
        writer.finish(&FINGERPRINT).unwrap();
        path
    }

    fn edit_header(path: &Path, edit: impl FnOnce(&mut Header)) {
        let mut bytes = std::fs::read(path).unwrap();
        edit(bytemuck::from_bytes_mut(&mut bytes[..size_of::<Header>()]));
        std::fs::write(path, &bytes).unwrap();
    }

    #[test]
    fn sections_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let reader = SectionReader::open(&write_sample(&dir), &FINGERPRINT).unwrap();
        let a: Slab<u32> = reader.section(A).unwrap();
        assert!(a.is_mapped());
        assert_eq!(&a[..], &(0..1000).collect::<Vec<_>>()[..]);
        assert_eq!(&reader.section::<u64>(B).unwrap()[..], &[1, 2, 3]);
        assert!(!reader.contains(3));
    }

    #[test]
    fn a_stale_index_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_sample(&dir);
        assert!(matches!(
            SectionReader::open(&path, &[0; FINGERPRINT_LEN]),
            Err(IndexError::Stale)
        ));
    }

    #[test]
    fn a_foreign_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not.nzidx");
        std::fs::write(&path, vec![0u8; SECTION_ALIGN]).unwrap();
        assert!(matches!(
            SectionReader::open(&path, &FINGERPRINT),
            Err(IndexError::NotAnIndex)
        ));
    }

    #[test]
    fn a_truncated_file_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_sample(&dir);
        // Cut into the records of the first section.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..SECTION_ALIGN + 100]).unwrap();
        let reader = SectionReader::open(&path, &FINGERPRINT).unwrap();
        assert!(matches!(
            reader.section::<u32>(A),
            Err(IndexError::Corrupt(_))
        ));
    }

    #[test]
    fn another_real_width_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_sample(&dir);
        edit_header(&path, |h| {
            h.real_width = if size_of::<Real>() == 4 { 8 } else { 4 }
        });
        assert!(matches!(
            SectionReader::open(&path, &FINGERPRINT),
            Err(IndexError::RealWidth(_))
        ));
    }

    #[test]
    fn a_section_of_the_wrong_stride_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let reader = SectionReader::open(&write_sample(&dir), &FINGERPRINT).unwrap();
        assert!(matches!(
            reader.section::<u64>(A),
            Err(IndexError::Corrupt(_))
        ));
    }

    #[test]
    fn sections_start_on_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        let reader = SectionReader::open(&write_sample(&dir), &FINGERPRINT).unwrap();
        for section in reader.header.sections.iter().filter(|s| s.tag != 0) {
            assert_eq!(
                section.offset as usize % SECTION_ALIGN,
                0,
                "tag {}",
                section.tag
            );
        }
    }
}
