//! The cache a decoder's calls attend to: storage a framework allocates from a `CacheSpec`, and
//! the views of it that the decoder's calls borrow.
//!
//! - A spec lists pools, each the tokens a block holds and the bytes a block of one layer spans
//!   (its page), and groups, each layers of one kind whose blocks come from one pool.
//! - A `Cache` holds `blocks` pages of every layer of every group, one allocation a layer.
//! - A cache serves the decoders whose spec it was allocated for. A decoder's first call builds
//!   its views of the layers; every later call borrows them, one call at a time.

use super::Decoder;
use furiosa_opt_std::prelude::{Device, HbmBuffer};
use std::any::Any;
use std::io;
use std::mem;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, PoisonError};

/// The cache a decoder's calls read: block formats, and the groups of layers that use them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSpec {
    /// Number of decoder layers covered by the groups.
    pub layer_count: usize,
    /// Block formats, which groups name by index.
    pub pools: Vec<Pool>,
    /// Engine storage groups. Groups with the same `logical_group` use one logical block table.
    pub groups: Vec<Group>,
}

/// A block format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pool {
    /// Tokens one block holds.
    pub block: usize,
    /// Bytes one block of one layer spans: a page.
    pub bytes: usize,
    /// Logical page shape and byte strides, independent of the physical allocation address.
    pub format: PageFormat,
}

/// The logical tensor format of one cache page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageFormat {
    /// Bytes occupied by one element.
    pub element_bytes: usize,
    /// Tensor dimensions, with the token-block dimension first.
    pub shape: Vec<usize>,
    /// Byte stride of each dimension.
    pub strides: Vec<usize>,
}

impl PageFormat {
    /// A contiguous row-major page format.
    pub fn contiguous(shape: Vec<usize>, element_bytes: usize) -> io::Result<Self> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if element_bytes == 0 || shape.is_empty() || shape.contains(&0) {
            return Err(invalid(
                "cache page format has empty dimensions or elements",
            ));
        }
        let mut stride = element_bytes;
        let mut strides = vec![0; shape.len()];
        for (dim, item) in shape.iter().zip(strides.iter_mut()).rev() {
            *item = stride;
            stride = stride
                .checked_mul(*dim)
                .ok_or_else(|| invalid("cache page format byte span overflows"))?;
        }
        Ok(Self {
            element_bytes,
            shape,
            strides,
        })
    }

    fn validate(&self, page_bytes: usize, block: usize) -> io::Result<()> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if self.element_bytes == 0
            || self.shape.is_empty()
            || self.shape.len() != self.strides.len()
        {
            return Err(invalid("cache page format has malformed dimensions"));
        }
        if self.shape[0] != block || self.shape.iter().any(|&dim| dim == 0) {
            return Err(invalid("cache page format does not match its token block"));
        }
        // Validate dimensions in physical stride order so axis permutations remain legal.
        let mut axes: Vec<_> = (0..self.shape.len()).collect();
        axes.sort_unstable_by_key(|&axis| self.strides[axis]);
        let mut extent = self.element_bytes;
        for axis in axes {
            let dim = self.shape[axis];
            let stride = self.strides[axis];
            if stride % self.element_bytes != 0 {
                return Err(invalid("cache page strides do not align to element size"));
            }
            if stride < extent {
                return Err(invalid("cache page strides overlap logical elements"));
            }
            extent = (dim - 1)
                .checked_mul(stride)
                .and_then(|span| span.checked_add(extent))
                .ok_or_else(|| invalid("cache page format byte span overflows"))?;
        }
        if extent != page_bytes {
            return Err(invalid("cache page format span differs from its pool size"));
        }
        Ok(())
    }
}

/// Layers whose cache has one kind and takes its blocks from one pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// Index of the pool in `CacheSpec::pools`.
    pub pool: usize,
    /// Stable model layer IDs held by this group; they need not be contiguous.
    pub layers: Vec<usize>,
    /// Logical cache group this engine storage group belongs to.
    pub logical_group: usize,
    pub kind: Kind,
}

/// What the blocks of a group hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Keys and values of positions, a block of positions each, in the request's block order.
    /// `window` limits how many prior positions each row attends; `None` attends the full history.
    Paged { window: Option<NonZeroUsize> },
    /// A recurrent state a block: the current one, then one for each draft position.
    State,
    /// Keys and values of `length` encoder positions, which a call writes once and later calls
    /// read.
    Cross { length: usize },
}

impl CacheSpec {
    /// The cache the calls of `model` read.
    pub fn of<D: Decoder>(model: &D) -> Self {
        model.limits().cache
    }

    /// Validate model-neutral layer, pool and logical-group metadata.
    pub fn validate(&self) -> io::Result<()> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if self.layer_count == 0 || self.groups.is_empty() {
            return Err(invalid("cache spec has no layers or groups"));
        }
        for pool in &self.pools {
            if pool.block == 0 || pool.bytes == 0 {
                return Err(invalid("cache pool has empty blocks"));
            }
            pool.format.validate(pool.bytes, pool.block)?;
        }
        let mut layer_owners = vec![None; self.layer_count];
        let mut logical_groups = std::collections::BTreeMap::new();
        for (index, group) in self.groups.iter().enumerate() {
            let pool = self
                .pools
                .get(group.pool)
                .ok_or_else(|| invalid("cache group names no pool"))?;
            if group.layers.is_empty() {
                return Err(invalid("cache group has no layers"));
            }
            for &layer in &group.layers {
                let owner = layer_owners
                    .get_mut(layer)
                    .ok_or_else(|| invalid("cache group names an unknown layer"))?;
                if owner.replace(index).is_some() {
                    return Err(invalid("cache layer appears in more than one group"));
                }
            }
            let semantics = (pool.block, group.kind);
            if logical_groups
                .insert(group.logical_group, semantics)
                .is_some_and(|previous| previous != semantics)
            {
                return Err(invalid(
                    "logical cache groups disagree on block size, attention kind, or window",
                ));
            }
        }
        if layer_owners.iter().any(Option::is_none) {
            return Err(invalid("cache spec does not cover every model layer"));
        }
        if logical_groups
            .keys()
            .copied()
            .enumerate()
            .any(|(expected, actual)| expected != actual)
        {
            return Err(invalid("logical cache group IDs are not contiguous"));
        }
        Ok(())
    }

    /// Validate metadata and reject cache kinds the decoder serving path cannot execute.
    pub fn validate_for_serving(&self) -> io::Result<()> {
        self.validate()?;
        if self
            .groups
            .iter()
            .any(|group| !matches!(group.kind, Kind::Paged { .. }))
        {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "decoder serving supports paged cache groups only",
            ));
        }
        Ok(())
    }

    /// Bytes of each layer's storage of `blocks` blocks. Refuses invalid metadata, no blocks, or
    /// a page whose base-relative byte offset passes the signed-I32 range.
    pub fn layer_sizes(&self, blocks: usize) -> io::Result<Vec<usize>> {
        self.validate()?;
        self.layer_sizes_validated(blocks)
    }

    fn layer_sizes_validated(&self, blocks: usize) -> io::Result<Vec<usize>> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if blocks == 0 {
            return Err(invalid("cache blocks must be positive"));
        }
        let mut sizes = vec![0; self.layer_count];
        for group in &self.groups {
            let pool = &self.pools[group.pool];
            // A device table holds a page's byte offset within its layer as an I32.
            let size = (blocks.checked_mul(pool.bytes))
                .filter(|&size| size <= 1 << 31)
                .ok_or_else(|| invalid("cache blocks pass I32 byte offsets"))?;
            for &layer in &group.layers {
                sizes[layer] = size;
            }
        }
        Ok(sizes)
    }
}

/// The storage of a spec, with the views one decoder keeps of it.
pub struct Cache {
    /// Each model layer's pages back to back, indexed by stable model layer ID.
    layers: Vec<HbmBuffer>,
    /// Blocks per layer.
    blocks: usize,
    /// The spec the layers were allocated for.
    spec: CacheSpec,
    views: Mutex<Views>,
}

/// The views a decoder keeps of a cache's layers.
enum Views {
    /// No call has built any.
    Empty,
    /// Built by a call, and free.
    Free(Box<dyn Any + Send>),
    /// Lent to a call.
    Lent,
}

impl Cache {
    /// `blocks` blocks of every layer `spec` lists, on `device`; block `j` of a layer is its bytes
    /// `j * page..(j + 1) * page`.
    pub fn new(device: &Device, spec: &CacheSpec, blocks: usize) -> io::Result<Self> {
        spec.validate_for_serving()?;
        let layers = (spec.layer_sizes_validated(blocks)?.into_iter())
            .map(|size| HbmBuffer::alloc(device, size).map_err(io::Error::other))
            .collect::<io::Result<_>>()?;
        Ok(Self {
            layers,
            blocks,
            spec: spec.clone(),
            views: Mutex::new(Views::Empty),
        })
    }

    /// Each layer's storage, indexed by stable model layer ID.
    pub fn layers(&self) -> &[HbmBuffer] {
        &self.layers
    }

    /// Blocks per layer.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// The spec the cache was allocated for.
    pub fn spec(&self) -> &CacheSpec {
        &self.spec
    }

    /// The views `build` makes of the layers, lent to one call at a time: the first call that asks
    /// for views of type `V` builds them, and they stay until a call asks for another type.
    pub(super) fn lend<V: Send + 'static>(
        &self,
        build: impl FnOnce(&[HbmBuffer]) -> io::Result<V>,
    ) -> io::Result<Lent<'_, V>> {
        let mut slot = self.views.lock().unwrap_or_else(PoisonError::into_inner);
        let views = match mem::replace(&mut *slot, Views::Lent) {
            Views::Lent => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cache is already borrowed",
                ));
            }
            Views::Free(views) => views.downcast::<V>().ok(),
            Views::Empty => None,
        };
        drop(slot);
        let mut lent = Lent { cache: self, views };
        if lent.views.is_none() {
            lent.views = Some(Box::new(build(&self.layers)?));
        }
        Ok(lent)
    }
}

/// A cache's views lent to one call; they return to the cache when the loan drops.
pub(super) struct Lent<'c, V: Send + 'static> {
    cache: &'c Cache,
    /// Present until the loan drops.
    views: Option<Box<V>>,
}

impl<V: Send + 'static> Deref for Lent<'_, V> {
    type Target = V;

    fn deref(&self) -> &V {
        (self.views.as_deref()).unwrap_or_else(|| unreachable!("a loan holds its views"))
    }
}

impl<V: Send + 'static> DerefMut for Lent<'_, V> {
    fn deref_mut(&mut self) -> &mut V {
        (self.views.as_deref_mut()).unwrap_or_else(|| unreachable!("a loan holds its views"))
    }
}

impl<V: Send + 'static> Drop for Lent<'_, V> {
    fn drop(&mut self) {
        let views = match self.views.take() {
            Some(views) => Views::Free(views),
            None => Views::Empty,
        };
        *self
            .cache
            .views
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = views;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    fn pool(block: usize, bytes: usize) -> Pool {
        Pool {
            block,
            bytes,
            format: PageFormat::contiguous(vec![block, bytes / block], 1).unwrap(),
        }
    }

    /// One pool of `bytes`-byte pages, one paged group over `layers`.
    fn paged(bytes: usize, layers: Range<usize>) -> CacheSpec {
        let layer_count = layers.end;
        CacheSpec {
            layer_count,
            pools: vec![pool(16, bytes)],
            groups: vec![Group {
                pool: 0,
                layers: layers.collect(),
                logical_group: 0,
                kind: Kind::Paged { window: None },
            }],
        }
    }

    /// A cache of no storage, for loans whose views read none.
    fn detached() -> Cache {
        Cache {
            layers: Vec::new(),
            blocks: 1,
            spec: CacheSpec {
                layer_count: 0,
                pools: vec![pool(16, 64)],
                groups: Vec::new(),
            },
            views: Mutex::new(Views::Empty),
        }
    }

    /// Every layer of every group holds a page of its pool for each block, group by group.
    #[test]
    fn sizes_each_layer_a_page_a_block() -> io::Result<()> {
        let mut spec = paged(64, 0..2);
        spec.layer_count = 3;
        spec.pools.push(pool(16, 32));
        spec.groups.push(Group {
            pool: 1,
            layers: vec![2],
            logical_group: 1,
            kind: Kind::State,
        });
        assert_eq!(spec.layer_sizes(3)?, [192, 192, 96]);
        Ok(())
    }

    /// Each rule refuses a cache that breaks only it, as invalid input with its own message.
    #[test]
    fn refuses_unallocatable_caches() {
        let mut orphan = paged(64, 0..1);
        orphan.groups[0].pool = 1;
        let mut tokenless = paged(64, 0..1);
        tokenless.pools[0].block = 0;
        let cases = [
            (paged(64, 0..1), 0, "cache blocks must be positive"),
            (orphan, 1, "cache group names no pool"),
            (paged(0, 0..1), 1, "cache pool has empty blocks"),
            (tokenless, 1, "cache pool has empty blocks"),
            (
                paged(64, 0..1),
                (1 << 25) + 1,
                "cache blocks pass I32 byte offsets",
            ),
            (
                paged(64, 0..1),
                usize::MAX,
                "cache blocks pass I32 byte offsets",
            ),
        ];
        for (spec, blocks, message) in cases {
            let error = spec.layer_sizes(blocks).expect_err(message);
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{message}");
            assert_eq!(error.to_string(), message);
        }
        assert!(paged(64, 0..1).layer_sizes(1 << 25).is_ok());
    }

    /// Descriptors cover every stable model layer once, and logical groups have one block size,
    /// attention kind, and window even when engine groups use different page formats.
    #[test]
    fn validates_layer_and_logical_group_mapping() {
        let mut spec = CacheSpec {
            layer_count: 3,
            pools: vec![pool(16, 64), pool(16, 32)],
            groups: vec![
                Group {
                    pool: 0,
                    layers: vec![0, 2],
                    logical_group: 0,
                    kind: Kind::Paged { window: None },
                },
                Group {
                    pool: 1,
                    layers: vec![1],
                    logical_group: 0,
                    kind: Kind::Paged { window: None },
                },
            ],
        };
        assert!(spec.validate().is_ok());

        spec.groups[1].layers = vec![2];
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache layer appears in more than one group"
        );
        spec.groups[1].layers = vec![1];
        spec.groups[1].kind = Kind::Paged {
            window: NonZeroUsize::new(8),
        };
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "logical cache groups disagree on block size, attention kind, or window"
        );
        spec.groups[1].kind = Kind::Paged { window: None };
        spec.groups[1].logical_group = 2;
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "logical cache group IDs are not contiguous"
        );
        spec.groups[1].logical_group = 0;
        spec.groups[1].layers.clear();
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache group has no layers"
        );
        spec.groups[1].layers = vec![1];
        spec.layer_count = 4;
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache spec does not cover every model layer"
        );
    }

    /// Page tensor metadata cannot alias elements or claim a span different from its pool page.
    #[test]
    fn validates_page_shapes_and_byte_strides() {
        let mut spec = paged(64, 0..1);
        spec.pools[0].format.shape = vec![16, 2];
        spec.pools[0].format.strides = vec![1, 1];
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache page strides overlap logical elements"
        );
        spec.pools[0].format = PageFormat {
            element_bytes: 2,
            shape: vec![16, 2],
            strides: vec![8, 2],
        };
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache page format span differs from its pool size"
        );
        spec.pools[0].format = PageFormat {
            element_bytes: 2,
            shape: vec![8, 8],
            strides: vec![16, 2],
        };
        assert_eq!(
            spec.validate().unwrap_err().to_string(),
            "cache page format does not match its token block"
        );
    }

    /// A valid cache page can permute its physical axes without aliasing its logical elements.
    #[test]
    fn validates_permuted_page_strides() {
        let shape = [4, 2, 3, 5];
        let orders = [[0, 1, 2, 3], [0, 2, 1, 3], [1, 0, 2, 3]];
        for order in orders {
            let mut strides = [0; 4];
            let mut stride = 2;
            for axis in order.into_iter().rev() {
                strides[axis] = stride;
                stride *= shape[axis];
            }
            let spec = CacheSpec {
                layer_count: 1,
                pools: vec![Pool {
                    block: shape[0],
                    bytes: stride,
                    format: PageFormat {
                        element_bytes: 2,
                        shape: shape.to_vec(),
                        strides: strides.to_vec(),
                    },
                }],
                groups: vec![Group {
                    pool: 0,
                    layers: vec![0],
                    logical_group: 0,
                    kind: Kind::Paged { window: None },
                }],
            };
            assert!(spec.validate().is_ok(), "order {order:?}");
        }
    }

    /// Recurrent state and encoder cache kinds are rejected before allocation or call setup.
    #[test]
    fn refuses_unserved_cache_kinds() {
        for kind in [Kind::State, Kind::Cross { length: 32 }] {
            let mut spec = paged(64, 0..1);
            spec.groups[0].kind = kind;
            let error = spec.validate_for_serving().unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::Unsupported);
            assert_eq!(
                error.to_string(),
                "decoder serving supports paged cache groups only"
            );
        }
    }

    /// The first loan of a type builds its views and later loans borrow them, one at a time; a
    /// loan of another type builds its own in their place.
    #[test]
    fn lends_its_views_to_one_call_at_a_time() -> io::Result<()> {
        let cache = detached();
        let mut lent = cache.lend(|_| Ok(7u8))?;
        *lent += 1;
        let error = cache
            .lend(|_| Ok(0u8))
            .err()
            .expect("a lent cache is refused");
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(error.to_string(), "cache is already borrowed");
        drop(lent);
        let built = |_: &[HbmBuffer]| -> io::Result<u8> { unreachable!("the views exist") };
        assert_eq!(*cache.lend(built)?, 8);
        assert_eq!(*cache.lend(|_| Ok(3u16))?, 3);
        assert_eq!(*cache.lend(|_| Ok(1u8))?, 1);
        Ok(())
    }
}
