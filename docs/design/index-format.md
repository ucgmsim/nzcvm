# The compiled mesh index

A mesh model is a tetrahedral mesh plus a bounding-volume hierarchy (BVH) over
its simplices. Until now every process that queried one read the mesh off disk
and ran the BVH build itself, then held the result on its heap. Two things
about that stopped scaling.

The models are large. A country-scale mesh runs to tens of millions of
simplices, and each worker process keeps its own full copy on its heap, so
memory use multiplies by the worker count.

The models couldn't leave the process. The BVH is a Rust structure with no
Python representation, so `--distributed` could only run thread workers inside
one interpreter, with a registry of live pointers in place of pickling. A
second node couldn't load the tree at all.

The index answers both. The build writes its output to a file whose layout is
the in-memory layout. A worker opens the file and maps it, and the pages a
query touches come in on demand. Processes on a node share those pages through
the page cache, and a node elsewhere pages them off the shared filesystem. No
process rebuilds the tree and no process copies it.

## What was already true

`MeshModel` already had the structure this needs. `Bvh::build_par` runs once
in `MeshModel::new`, `CompactBvh::from_bvh` converts the crate's tree into
packed 56-byte nodes and drops the original, and after that a query walks a
`&[CompactNode]` and a `&[Simplex]` without touching the `bvh` crate. The one
query-time use of the crate left is the outer 4-D tree over the meshes, which
has one entry per mesh and rebuilds in microseconds.

So the tree didn't need an algorithmic change. The work was making the slices
come from a file.

## Records

Each hot array is plain-old-data with fixed-width fields and no pointers.

| Record        | Size (f32) | Contents                                                       |
|---------------|-----------:|----------------------------------------------------------------|
| `CompactNode` |       56 B | two children, each an axis-aligned bounding box (AABB) and a packed child slot |
| `Simplex`     |       48 B | anchor vertex and the inverse edge matrix                       |
| `VertexRefs`  |       16 B | four quality indices, or 4 B for a constant model               |
| `Quality`     |       24 B | six components per mesh vertex                                  |

Each record is `#[repr(C)]` and derives `zerocopy`'s `FromBytes`, `IntoBytes`,
`KnownLayout`, and `Immutable`. That pins the layout and lets a byte range
become a `&[T]` with size and alignment checked, with no `unsafe` in the
project's code. `f32` and `u32` have no invalid bit patterns, which is what
makes `FromBytes` sound for them.

Two of the records changed to get there. `Simplex` holds `[Real; 3]` and
`[[Real; 3]; 3]` rather than nalgebra's `Point3` and `Matrix3`, because the
derives can't see through a foreign type, and `Matrix3::from` on the array is a
load the optimiser folds away. `ChildRef` holds `min`, `max`, and a raw `u32`
slot rather than an `Aabb` and a `ChildSlot`, for the same reason. With
`Real = f64` the slot would end the record on a 4-byte field and pick up tail
padding, so that build adds an explicit `_pad` field.

## `Slab<T>`

```rust
pub enum Slab<T: Record> {
    Owned(Vec<T>),
    Mapped { mmap: Arc<Mmap>, offset: usize, len: usize },
}
```

`Slab` dereferences to `&[T]` in either variant. `MeshModel`'s four
arrays are slabs. The build produces `Owned`, and `index::open` produces
`Mapped`, sharing one `Arc<Mmap>` across the sections of a file. Nothing
downstream changed, because everything already took a slice.

The cast in `Deref` can't fail. `Slab::mapped` proved the exact range castable
when it built the slab, and nothing modifies the mapping or the range after
that.

## The file

```text
page 0        Header, then the model name as UTF-8
nodes         [CompactNode]   the tree, in the order the build emitted it
simplices     [Simplex]       in BVH leaf order
refs          [VertexRefs] or [u32], by model kind
kinds         [u32]           only for a mixed model map
qualities     [Quality]       one per mesh vertex
```

Each section starts on a 4 KiB boundary, which aligns it for its record type
and for the mapping. The header records the byte offset and record count of
each section, the `Real` width, the layout version, the model's bounding box,
priority, transform, and name, and a 32-byte fingerprint of the source mesh
supplied by the caller. A reader built with the other `Real` width refuses the
file rather than misread it. Numbers are little-endian, and the file says
nothing about byte order.

The writer puts the file down under a `.partial` name and renames it into
place, so a crash mid-write leaves nothing that parses.

The index goes in the same directory as the mesh: `models/Wellington.zarr` compiles to
`models/Wellington.nzidx`. The zarr stays the source of truth, and the index
is a derived cache.

### Why not inside the zarr

Zarr was the first thought, since the mesh is already stored there. Two things
ruled it out. `zarr-python` 3.3 rejects a structured dtype, so an index array
could only be a `uint8` blob, and a blob describes none of its contents. And
mapping it would mean reaching past the library to the chunk file at `c/0`,
which ties the format to zarr's key layout for no return. A flat file with its
own header is the same bytes without the ceremony.

## Freshness

An index is only right for the mesh it came from. A content hash would mean
reading the arrays, which defeats the point, so the fingerprint is instead a
`blake2b` over the path, size, and modification time of every file in the
store, plus the bytes of each `zarr.json`. Checking it costs a directory walk.

That errs towards rebuilding. A mesh rewritten with identical contents fails to
match its old index, the loader builds in memory as before, and
`nzcvm index build` refreshes the file.

## Loading

`MeshModel.from_path` maps the index when one exists and its fingerprint
matches, and builds otherwise. `ModelTree.load_models` didn't change. The
outer tree over the meshes still comes from the `bvh` crate at open, over the
bounding boxes in their headers.

`nzcvm index build models/*.zarr` compiles every mesh, and the `models` recipe
runs it after the basin meshes finish.

## Network filesystems

The design assumed local disk and Lustre, and Cascade mounts the Network File
System (NFS). That changes cold start and the failure modes rather than steady
state.

On a local disk, servicing a page fault takes tens of microseconds. On NFS a fault that
misses the client's page cache is a `READ` remote procedure call (RPC), on the
order of a millisecond, fetching at most `rsize` (typically 1 MiB) plus
whatever readahead the client adds. Sequential access runs at link speed.
Scattered access pays a round trip per page. The failure modes have no local
equivalent either. An I/O error, or `ESTALE` after the server unlinks the
file, arrives as `SIGBUS` on the faulting instruction, and the process ends
with no exception to catch.

Steady state isn't the concern. Once pages are in the client cache, repeated
queries run at memory speed, and every worker process on the node shares them.

### The access pattern

`CompactBvh::from_bvh` emits nodes in depth-first preorder, so a subtree is
contiguous in the `nodes` section, and simplices sit in leaf order, so the
simplices of a subtree are contiguous too. The top of the tree occupies the
first pages of the section, and every query shares it. A `generate` chunk is
spatially coherent, so its queries descend the same few subtrees, and its
working set is a compact run of pages. An isolated borehole is the worst case,
at roughly `log n` scattered pages, and also the case where twenty round trips
is still only forty milliseconds. The expensive scenario is the first chunk of
a full-volume run on a cold client, faulting most of the tree in one RPC at a
time.

### The policy to add

`memmap2` exposes `populate()`, `advise()`, and `advise_range()` with
`WillNeed`, `Sequential`, `Random`, and `PopulateRead`. The loader should
choose by filesystem type at open, from `statfs`:

| Situation      | Local or Lustre                     | NFS                                                  |
|----------------|-------------------------------------|------------------------------------------------------|
| At open        | `WillNeed` on nodes and simplices   | `PopulateRead` index                    |
| During queries | `Normal`                            | `Normal`                                             |
| `Random`       | never                               | never, since it disables readahead and leaves one bare RPC per fault |
| Fallback       |                                     | read into `Slab::Owned`, which gives up page sharing and rules out `SIGBUS` |

`PopulateRead` on NFS is one sequential read at link speed. A 2 GB index
takes a second or two, paid once per node, because later processes hit the
populated cache. Populated pages remain evictable, so refaults can still happen under
memory pressure. `mlock` exists behind rlimits if that ever matters.

### Naming, because of `ESTALE`

`rename` over an existing index is atomic, but a client with the old inode
mapped gets `ESTALE`, and so `SIGBUS`, on its next fault. Recompiling an index
under a running job ends that job. The fix is immutable, content-addressed
index files, `Wellington.<hash>.nzidx`, so a recompile writes a new file and
never touches an old one, with removal of superseded indexes as a separate
step run when nothing is using them. The present version uses a fixed name and
relies on nobody recompiling under a live job.

### Layout adjustments worth making

Section alignment of 1 MiB rather than 4 KiB, matching `rsize`, so a section's
prefetch never straddles into the next. That wastes under 5 MiB per file. Hot
sections first, which the layout already does. A header page that a loader can
validate for 45 meshes with 45 small reads before mapping anything, which it
also already does.

### To measure on Cascade rather than assume

Cold open plus first chunk against warm, on NFS and on local scratch, with
`nfsstat -c` before and after to count `READ` RPCs. If lazy `WillNeed` on NFS
comes within a few percent of `PopulateRead`, prefer lazy, since it's the
friendlier default for boreholes. Job-start load matters too: thirty nodes
each pulling 2 GB from one NFS server is 60 GB with no striping to spread it.
If that hurts, the fix is to stagger the starts or to stage the files to
node-local scratch, and the format works unchanged with either.

The intended surface is `--index-prefetch {auto,lazy,populate,read}`, with
`auto` following the preceding table, so that the measurement is a flag flip.

## Is the build worth skipping at all?

The honest question before any of this: does the BVH build cost enough to
matter? `benchmarks/benchmark_index.py` builds a synthetic tomography mesh of
a chosen size and times both paths. At 2.8 million tetrahedra on one
development machine:

| Step                                              | Time                          |
|---------------------------------------------------|------------------------------:|
| read zarr and build the BVH                       | 1.0 to 1.5 s                  |
| open the index (header, `mmap`, fingerprint walk) | 7 to 13 ms                    |
| index on disk                                     | 252 MB, 91 B per tetrahedron  |
| query 200,000 points, built tree                  | 130 ms                        |
| query 200,000 points, mapped, first pass          | 163 ms                        |
| query 200,000 points, mapped, page cache warm     | 146 ms                        |

Build time scales close to linearly, so a 20-million-tetrahedron mesh costs on
the order of ten seconds per process that loads it. A `--distributed` run with
eight workers on a node spends eighty CPU-seconds and eight copies of memory
arriving at the same tree. Against that, the index costs about the size of the
source mesh again on disk.

The mapped tree answered queries about 12% slower than the built one with the
page cache warm, and 25% slower on the first pass while pages faulted in. Both
trees run the same code over the same record layout, so the gap is the memory
itself: file-backed pages can't use transparent huge pages, so a scattered tree
walk misses the translation lookaside buffer more often than it does over an
anonymous `Vec`. That's
the price of sharing the pages, and the first-pass figure is the one the NFS
section is about.

## Not done yet

- Pickling by path and removal of `nzcvm/registry.py`, so `--distributed` runs
  worker processes. A mapped `MeshModel` reduces to its index path, so the
  remaining work is the plumbing.
- The `Surface` and `Coastline` indexes, the same format family with 2-D
  sections. `NZ_DEM_HD` is large enough to matter.
- The NFS policy and the content-addressed naming described in the preceding
  sections.
- `Slab` reporting resident rather than mapped pages in `deep_size_of`, once
  there is a reason to distinguish them.
