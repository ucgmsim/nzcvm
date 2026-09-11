//! A memory-compact, query-only BVH over a mesh's simplices.
//!
//! The tree is built with the `bvh` crate (parallel SAH build) and then
//! converted into this representation, which is what the mesh actually keeps:
//!
//! * only *internal* nodes are stored — leaves are encoded as [`ChildSlot`]s
//!   pointing directly into the simplex array;
//! * child indices are `u32` instead of `usize`, and the never-queried
//!   `parent_index` is dropped;
//! * subtrees of up to [`MAX_LEAF_SIZE`] simplices are collapsed into a single
//!   leaf, and the simplices are reordered so each leaf's simplices are
//!   contiguous in memory.
//!
//! With `Real = f32` a [`CompactNode`] is 56 bytes and there are at most
//! `⌈n / 2⌉` of them, compared to `2n - 1` nodes of 80 bytes each for the
//! `bvh` crate's tree.

use bitfield_struct::bitfield;
use bvh::aabb::Aabb;
use bvh::bvh::{Bvh, BvhNode};
use deepsize::{Context, DeepSizeOf};
use nalgebra::Point3;
use smallvec::SmallVec;

use crate::index::{IndexError, SectionReader, SectionWriter, tag};
use crate::real::Real;
use crate::simplex::Simplex;
use crate::slab::Slab;
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Maximum number of simplices collapsed into a single leaf.
///
/// Simplices within a leaf are contiguous, so testing them is a linear scan
/// of adjacent 48-byte records — cheap compared to extra node traversal.
pub const MAX_LEAF_SIZE: u32 = 4;

/// Largest simplex index a [`ChildSlot`] payload can hold.
const MAX_START: u32 = (1 << 29) - 1;

const STACK_CAPACITY: usize = 16;

/// A contiguous run of simplices forming one leaf.
#[derive(Clone, Copy)]
struct LeafRange {
    start: u32,
    count: u32,
}

impl LeafRange {
    fn end(self) -> u32 {
        self.start + self.count
    }
}

/// What a [`ChildSlot`] points at.
#[derive(Clone, Copy)]
enum Child {
    /// Index into [`CompactBvh::nodes`].
    Node(u32),
    Leaf(LeafRange),
}

/// A [`Child`] packed into 4 bytes.
///
/// `Child` itself is 8 bytes — a `u32` payload leaves no niche for the
/// discriminant — so the tag, the leaf size and the payload share one word.
/// The layout is declared rather than hand-shifted; [`ChildSlot::unpack`]
/// recovers a [`Child`] on the stack and the optimiser folds it away.
#[bitfield(u32)]
struct ChildSlot {
    /// Node index, or the first simplex of a leaf.  29 bits limits a single
    /// mesh to 2^29 (~537 M) simplices.
    #[bits(29)]
    payload: u32,
    /// Simplices in the leaf, `1..=`[`MAX_LEAF_SIZE`]; unused for nodes.
    #[bits(2, default = 1, from = count_from_bits, into = count_into_bits)]
    count: u32,
    is_leaf: bool,
}

const fn count_from_bits(bits: u8) -> u32 {
    bits as u32 + 1
}

const fn count_into_bits(count: u32) -> u8 {
    (count - 1) as u8
}

impl ChildSlot {
    fn node(index: u32) -> Self {
        Self::new().with_payload(index)
    }

    fn leaf(range: LeafRange) -> Self {
        debug_assert!((1..=MAX_LEAF_SIZE).contains(&range.count));
        Self::new()
            .with_is_leaf(true)
            .with_payload(range.start)
            .with_count(range.count)
    }

    fn unpack(self) -> Child {
        if self.is_leaf() {
            Child::Leaf(LeafRange {
                start: self.payload(),
                count: self.count(),
            })
        } else {
            Child::Node(self.payload())
        }
    }
}

/// One child of an internal node: its bounding box and where it lives.
///
/// Plain arrays and a raw `u32` rather than `Aabb` and `ChildSlot`, because
/// this record is also the on-disk format and `zerocopy` has to be able to
/// cast it.  [`ChildRef::slot`] unpacks the slot on demand.
#[derive(Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct ChildRef {
    min: [Real; 3],
    max: [Real; 3],
    slot: u32,
    /// With `Real = f64` the record would otherwise end on a 4-byte field
    /// and pick up implicit tail padding, which the on-disk cast forbids.
    #[cfg(feature = "high_precision")]
    _pad: u32,
}

impl ChildRef {
    fn new(aabb: &Aabb<Real, 3>, slot: ChildSlot) -> Self {
        Self {
            min: aabb.min.into(),
            max: aabb.max.into(),
            slot: slot.into_bits(),
            #[cfg(feature = "high_precision")]
            _pad: 0,
        }
    }

    #[inline(always)]
    fn slot(&self) -> ChildSlot {
        ChildSlot::from_bits(self.slot)
    }

    /// The child's bounds, for the same containment test the rest of the
    /// crate uses.  Building the `Aabb` is two loads the optimiser folds away.
    #[inline(always)]
    fn contains(&self, p: &Point3<Real>) -> bool {
        Aabb::with_bounds(Point3::from(self.min), Point3::from(self.max)).contains(p)
    }
}

/// One internal node: both of its children.
#[derive(Clone, Copy, FromBytes, IntoBytes, KnownLayout, Immutable)]
#[repr(C)]
pub struct CompactNode {
    left: ChildRef,
    right: ChildRef,
}

// Each child is six reals of corners and one real's worth of slot: the `f64`
// build spells out the pad that `repr(C)` would otherwise add silently, so
// the record is exactly as wide at either width and the file layout is pinned.
const _: () = assert!(size_of::<CompactNode>() == 14 * size_of::<Real>());

pub struct CompactBvh {
    nodes: Slab<CompactNode>,
    /// The root child slot (a leaf for meshes of ≤ `MAX_LEAF_SIZE`
    /// simplices), or `None` for an empty mesh.
    root: Option<ChildSlot>,
}

impl DeepSizeOf for CompactBvh {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.nodes.deep_size_of_children(context)
    }
}

/// Number of shapes in each subtree of the source tree, indexed by node.
fn subtree_counts(nodes: &[BvhNode<Real, 3>]) -> Vec<u32> {
    fn count(nodes: &[BvhNode<Real, 3>], counts: &mut [u32], i: usize) -> u32 {
        let c = match nodes[i] {
            BvhNode::Leaf { .. } => 1,
            BvhNode::Node {
                child_l_index,
                child_r_index,
                ..
            } => count(nodes, counts, child_l_index) + count(nodes, counts, child_r_index),
        };
        counts[i] = c;
        c
    }
    let mut counts = vec![0u32; nodes.len()];
    if !nodes.is_empty() {
        count(nodes, &mut counts, 0);
    }
    counts
}

/// Append the shape indices of every leaf under `i` to `order`.
fn collect_shapes(nodes: &[BvhNode<Real, 3>], order: &mut Vec<u32>, i: usize) {
    match nodes[i] {
        BvhNode::Leaf { shape_index, .. } => order.push(shape_index as u32),
        BvhNode::Node {
            child_l_index,
            child_r_index,
            ..
        } => {
            collect_shapes(nodes, order, child_l_index);
            collect_shapes(nodes, order, child_r_index);
        }
    }
}

impl CompactBvh {
    /// Convert a freshly built `bvh` crate tree over `num_shapes` shapes.
    ///
    /// Returns the compact tree together with the leaf-order permutation:
    /// `order[new_index] = old_shape_index`.  The caller must gather its
    /// per-simplex arrays through `order` so that leaf slots index correctly.
    pub fn from_bvh(tree: &Bvh<Real, 3>, num_shapes: usize) -> (Self, Vec<u32>) {
        assert!(
            num_shapes <= MAX_START as usize,
            "mesh exceeds the compact BVH limit of 2^29 simplices"
        );
        if num_shapes == 0 {
            return (
                Self {
                    nodes: Slab::Owned(Vec::new()),
                    root: None,
                },
                Vec::new(),
            );
        }

        let counts = subtree_counts(&tree.nodes);
        let mut nodes = Vec::with_capacity(num_shapes / 2);
        let mut order = Vec::with_capacity(num_shapes);

        fn emit(
            src: &[BvhNode<Real, 3>],
            counts: &[u32],
            nodes: &mut Vec<CompactNode>,
            order: &mut Vec<u32>,
            i: usize,
        ) -> ChildSlot {
            if counts[i] <= MAX_LEAF_SIZE {
                let start = order.len() as u32;
                collect_shapes(src, order, i);
                return ChildSlot::leaf(LeafRange {
                    start,
                    count: counts[i],
                });
            }
            // A subtree holding more than one shape is always a `Node`.
            let BvhNode::Node {
                child_l_index,
                child_l_aabb,
                child_r_index,
                child_r_aabb,
                ..
            } = src[i]
            else {
                unreachable!("leaf node with subtree count > 1")
            };
            // The children's slots are only known once they have been
            // emitted, so reserve this node first and patch them in after.
            let index = nodes.len() as u32;
            let placeholder = ChildSlot::node(0);
            nodes.push(CompactNode {
                left: ChildRef::new(&child_l_aabb, placeholder),
                right: ChildRef::new(&child_r_aabb, placeholder),
            });
            let left = emit(src, counts, nodes, order, child_l_index);
            let right = emit(src, counts, nodes, order, child_r_index);
            nodes[index as usize].left.slot = left.into_bits();
            nodes[index as usize].right.slot = right.into_bits();
            ChildSlot::node(index)
        }

        let root = emit(&tree.nodes, &counts, &mut nodes, &mut order, 0);
        nodes.shrink_to_fit();
        (
            Self {
                nodes: Slab::Owned(nodes),
                root: Some(root),
            },
            order,
        )
    }

    /// The packed root slot, or `None` for an empty mesh.
    pub(crate) fn root_slot(&self) -> Option<u32> {
        self.root.map(ChildSlot::into_bits)
    }

    /// Append the node records to an index being written.
    pub(crate) fn write_sections(&self, writer: &mut SectionWriter) -> Result<(), IndexError> {
        writer.write(tag::NODES, &self.nodes)
    }

    /// Map the node records back out of an opened index.
    pub(crate) fn read_sections(reader: &SectionReader) -> Result<Self, IndexError> {
        Ok(Self {
            nodes: reader.section(tag::NODES)?,
            root: reader.root_slot().map(ChildSlot::from_bits),
        })
    }

    /// Iterator over the indices of all simplices that contain `point`.
    pub fn containing_indices<'a>(
        &'a self,
        simplices: &'a [Simplex],
        point: Point3<Real>,
    ) -> ContainingIndices<'a> {
        let mut stack = SmallVec::new();
        if let Some(root) = self.root {
            stack.push(root);
        }
        ContainingIndices {
            nodes: &self.nodes[..],
            simplices,
            point,
            stack,
            cursor: 0,
            end: 0,
        }
    }

    /// Maximum depth of the tree (0 when the root is a single leaf).
    pub fn depth(&self) -> usize {
        fn go(nodes: &[CompactNode], slot: ChildSlot) -> usize {
            match slot.unpack() {
                Child::Leaf(_) => 0,
                Child::Node(index) => {
                    let node = &nodes[index as usize];
                    1 + go(nodes, node.left.slot()).max(go(nodes, node.right.slot()))
                }
            }
        }
        self.root.map_or(0, |root| go(&self.nodes[..], root))
    }
}

/// See [`CompactBvh::containing_indices`].
pub struct ContainingIndices<'a> {
    nodes: &'a [CompactNode],
    simplices: &'a [Simplex],
    point: Point3<Real>,
    stack: SmallVec<[ChildSlot; STACK_CAPACITY]>,
    /// Range of the leaf currently being scanned.
    cursor: u32,
    end: u32,
}

impl Iterator for ContainingIndices<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            while self.cursor < self.end {
                let i = self.cursor as usize;
                self.cursor += 1;
                if self.simplices[i].contains(&self.point) {
                    return Some(i);
                }
            }
            match self.stack.pop()?.unpack() {
                Child::Leaf(range) => {
                    self.cursor = range.start;
                    self.end = range.end();
                }
                Child::Node(index) => {
                    let node = &self.nodes[index as usize];
                    if node.left.contains(&self.point) {
                        self.stack.push(node.left.slot());
                    }
                    if node.right.contains(&self.point) {
                        self.stack.push(node.right.slot());
                    }
                }
            }
        }
    }
}
