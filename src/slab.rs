//! Storage for a mesh's per-record arrays: owned in memory, or mapped from a
//! file.
//!
//! Every hot array in a [`MeshModel`](crate::mesh::MeshModel) is a run of
//! fixed-width plain-old-data records, so the same bytes serve as an in-memory
//! `Vec<T>` and as a region of a memory-mapped index file.  A [`Slab`] is one
//! such array behind a `&[T]` view, and nothing downstream needs to know which
//! kind it is.
//!
//! The mapped form casts bytes to records with `zerocopy`, which checks size
//! and alignment when the slab is created.  The record types opt in by
//! deriving [`FromBytes`], [`IntoBytes`], [`KnownLayout`] and [`Immutable`]
//! on a `#[repr(C)]` layout, which is also what pins the on-disk format.

use std::ops::Deref;
use std::sync::Arc;

use deepsize::{Context, DeepSizeOf};
use memmap2::Mmap;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// A record type a [`Slab`] can hold.
pub trait Record: FromBytes + IntoBytes + KnownLayout + Immutable + Copy {}

impl<T: FromBytes + IntoBytes + KnownLayout + Immutable + Copy> Record for T {}

/// An array of records, owned or memory-mapped.
pub enum Slab<T: Record> {
    Owned(Vec<T>),
    Mapped {
        mmap: Arc<Mmap>,
        /// Byte offset of the first record within the mapping.
        offset: usize,
        /// Number of records.
        len: usize,
    },
}

/// Why a byte range could not be viewed as records.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlabError {
    /// The range runs past the end of the mapping.
    OutOfBounds,
    /// The range does not start on a boundary the record type needs.
    Misaligned,
}

impl std::fmt::Display for SlabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlabError::OutOfBounds => write!(f, "record range runs past the end of the mapping"),
            SlabError::Misaligned => write!(f, "record range is misaligned for its record type"),
        }
    }
}

impl<T: Record> Slab<T> {
    /// View `len` records starting `offset` bytes into `mmap`.
    ///
    /// Size and alignment are checked here, once, so that [`Deref`] can stay
    /// infallible.
    pub fn mapped(mmap: Arc<Mmap>, offset: usize, len: usize) -> Result<Self, SlabError> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .and_then(|n| offset.checked_add(n))
            .filter(|end| *end <= mmap.len())
            .map(|end| &mmap[offset..end])
            .ok_or(SlabError::OutOfBounds)?;
        <[T]>::ref_from_bytes_with_elems(bytes, len).map_err(|_| SlabError::Misaligned)?;
        Ok(Slab::Mapped { mmap, offset, len })
    }

    pub fn len(&self) -> usize {
        match self {
            Slab::Owned(v) => v.len(),
            Slab::Mapped { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the records live in a file mapping rather than the heap.
    pub fn is_mapped(&self) -> bool {
        matches!(self, Slab::Mapped { .. })
    }

    /// Size of the records in bytes.
    pub fn bytes(&self) -> usize {
        self.len() * size_of::<T>()
    }
}

impl<T: Record> Deref for Slab<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        match self {
            Slab::Owned(v) => v,
            Slab::Mapped { mmap, offset, len } => {
                let bytes = &mmap[*offset..*offset + *len * size_of::<T>()];
                // `Slab::mapped` proved this exact range castable when the
                // slab was built, and neither the mapping nor the range has
                // changed since.
                <[T]>::ref_from_bytes_with_elems(bytes, *len)
                    .expect("slab range was validated at construction")
            }
        }
    }
}

impl<T: Record> From<Vec<T>> for Slab<T> {
    fn from(v: Vec<T>) -> Self {
        Slab::Owned(v)
    }
}

impl<T: Record> DeepSizeOf for Slab<T> {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        match self {
            Slab::Owned(v) => v.capacity() * size_of::<T>(),
            // Mapped pages are the kernel's to keep or drop, but they are the
            // memory a query touches, so report them as the model's size.
            Slab::Mapped { .. } => self.bytes(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use memmap2::MmapOptions;
    use std::io::Write;

    fn mapping_of(bytes: &[u8]) -> Arc<Mmap> {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(bytes).unwrap();
        // SAFETY: the file is a private temporary that nothing else writes.
        Arc::new(unsafe { MmapOptions::new().map(&file).unwrap() })
    }

    #[test]
    fn mapped_slab_reads_back_the_records() {
        let values: Vec<u32> = (0..10).collect();
        let mmap = mapping_of(values.as_bytes());
        let slab = Slab::<u32>::mapped(mmap, 0, 10).unwrap();
        assert!(slab.is_mapped());
        assert_eq!(&slab[..], &values[..]);
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        let mmap = mapping_of(&[0u8; 16]);
        assert_eq!(
            Slab::<u32>::mapped(mmap, 8, 3).err(),
            Some(SlabError::OutOfBounds)
        );
    }

    #[test]
    fn a_misaligned_range_is_refused() {
        let mmap = mapping_of(&[0u8; 16]);
        assert_eq!(
            Slab::<u32>::mapped(mmap, 1, 2).err(),
            Some(SlabError::Misaligned)
        );
    }

    #[test]
    fn owned_and_mapped_report_the_same_bytes() {
        let values: Vec<u32> = (0..10).collect();
        let owned = Slab::Owned(values.clone());
        let mapped = Slab::<u32>::mapped(mapping_of(values.as_bytes()), 0, 10).unwrap();
        assert_eq!(owned.bytes(), mapped.bytes());
        assert_eq!(owned.len(), mapped.len());
    }
}
