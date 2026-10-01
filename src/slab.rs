//! Storage for a mesh's per-record arrays: owned in memory, or mapped from a
//! file.
//!
//! Every hot array in a [`MeshModel`](crate::mesh::MeshModel) is a run of
//! fixed-width plain-old-data records, so the same bytes serve as an in-memory
//! `Vec<T>` and as a region of a memory-mapped index file.  A [`Slab`] is one
//! such array behind a `&[T]` view, and nothing downstream needs to know which
//! kind it is.
//!
//! The mapped form casts bytes to records with `bytemuck`, which checks size
//! and alignment once, when the slab is created.  A record type qualifies by
//! deriving [`Pod`](bytemuck::Pod) on a `#[repr(C)]` layout, which is also
//! what pins the on-disk format.

use std::ops::Deref;
use std::sync::Arc;

use deepsize::{Context, DeepSizeOf};
use memmap2::Mmap;

/// A record type a [`Slab`] can hold.
pub trait Record: bytemuck::Pod {}

impl<T: bytemuck::Pod> Record for T {}

/// An array of records, owned or memory-mapped.
pub struct Slab<T: Record>(Storage<T>);

enum Storage<T: 'static> {
    Owned(Vec<T>),
    Mapped {
        /// The checked view into `_mmap`.  `'static` stands in for "as long
        /// as `_mmap` lives", which the private field guarantees.
        records: &'static [T],
        _mmap: Arc<Mmap>,
    },
}

impl<T: Record> Slab<T> {
    /// View `len` records starting `offset` bytes into `mmap`, or `None` if
    /// that range runs past the end of the mapping or is misaligned for `T`.
    pub fn mapped(mmap: Arc<Mmap>, offset: usize, len: usize) -> Option<Self> {
        let end = len.checked_mul(size_of::<T>())?.checked_add(offset)?;
        let records: &[T] = bytemuck::try_cast_slice(mmap.get(offset..end)?).ok()?;
        // SAFETY: `records` borrows from the mapping, which the slab keeps
        // alive in `_mmap` and which is never remapped or written through.
        let records = unsafe { std::mem::transmute::<&[T], &'static [T]>(records) };
        Some(Slab(Storage::Mapped {
            records,
            _mmap: mmap,
        }))
    }

    /// Whether the records live in a file mapping rather than the heap.
    pub fn is_mapped(&self) -> bool {
        matches!(self.0, Storage::Mapped { .. })
    }
}

impl<T: Record> From<Vec<T>> for Slab<T> {
    fn from(v: Vec<T>) -> Self {
        Slab(Storage::Owned(v))
    }
}

impl<T: Record> Deref for Slab<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        match &self.0 {
            Storage::Owned(v) => v,
            Storage::Mapped { records, .. } => records,
        }
    }
}

impl<T: Record> DeepSizeOf for Slab<T> {
    fn deep_size_of_children(&self, _context: &mut Context) -> usize {
        match &self.0 {
            Storage::Owned(v) => v.capacity() * size_of::<T>(),
            // Mapped pages are the kernel's to keep or drop, but they are the
            // memory a query touches, so report them as the model's size.
            Storage::Mapped { records, .. } => size_of_val(*records),
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
        let mmap = mapping_of(bytemuck::cast_slice(&values));
        let slab = Slab::<u32>::mapped(mmap, 0, 10).unwrap();
        assert!(slab.is_mapped());
        assert_eq!(&slab[..], &values[..]);
    }

    #[test]
    fn a_range_past_the_end_is_refused() {
        let mmap = mapping_of(&[0u8; 16]);
        assert!(Slab::<u32>::mapped(mmap, 8, 3).is_none());
    }

    #[test]
    fn a_misaligned_range_is_refused() {
        let mmap = mapping_of(&[0u8; 16]);
        assert!(Slab::<u32>::mapped(mmap, 1, 2).is_none());
    }

    #[test]
    fn owned_and_mapped_report_the_same_size() {
        let values: Vec<u32> = (0..10).collect();
        let owned = Slab::from(values.clone());
        let mapped = Slab::<u32>::mapped(mapping_of(bytemuck::cast_slice(&values)), 0, 10).unwrap();
        assert_eq!(owned.len(), mapped.len());
        assert_eq!(owned.deep_size_of(), mapped.deep_size_of());
    }
}
