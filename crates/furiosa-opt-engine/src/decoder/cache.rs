//! The cache a decoder's calls attend to: storage a framework allocates from a `CacheSpec`, and
//! the views of it that the decoder's calls borrow.
//!
//! - A spec lists pools, each the tokens a block holds and the bytes a block of one layer spans
//!   (its page), and groups, each layers of one kind whose blocks come from one pool.
//! - A `Cache` holds `blocks` pages of every layer of every group, one allocation a layer.
//! - A cache serves the decoders whose spec it was allocated for. A decoder's first call builds
//!   its views of the layers; every later call borrows them, one call at a time.

use super::Decoder;
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_std::prelude::{Device, HbmBuffer};
use std::any::Any;
use std::io;
use std::iter;
use std::mem;
use std::ops::{Deref, DerefMut, Range};
use std::sync::{Mutex, PoisonError};

/// The cache a decoder's calls read: block formats, and the groups of layers that use them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheSpec {
    /// Block formats, which groups name by index.
    pub pools: Vec<Pool>,
    /// The cached layers, a kind and a pool each; a request names its blocks of every group.
    pub groups: Vec<Group>,
}

/// A block format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pool {
    /// Tokens one block holds.
    pub block: usize,
    /// Bytes one block of one layer spans: a page.
    pub bytes: usize,
}

/// Layers whose cache has one kind and takes its blocks from one pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    /// Index of the pool in `CacheSpec::pools`.
    pub pool: usize,
    /// The model layers the group caches.
    pub layers: Range<usize>,
    pub kind: Kind,
    /// Positions a row attends back to in a sliding window; `None` attends the whole history.
    pub window: Option<usize>,
}

/// What the blocks of a group hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Keys and values of positions, a block of positions each, in the request's block order.
    Paged,
    /// A recurrent state a block: the current one, then one for each draft position.
    State,
    /// Keys and values of `length` encoder positions, which a call writes once and later calls
    /// read.
    Cross { length: usize },
}

impl CacheSpec {
    /// The cache the calls of `model` read, however their rows' attention is tiled.
    pub fn of<D: Decoder>(model: &D) -> Self {
        model.limits(Tiling::default()).cache
    }

    /// Bytes of each layer's storage of `blocks` blocks, group by group. Refuses no blocks, a
    /// group of no pool, a pool of empty blocks, and a page whose byte offset passes I32.
    fn sizes(&self, blocks: usize) -> io::Result<Vec<usize>> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if blocks == 0 {
            return Err(invalid("cache blocks must be positive"));
        }
        let groups = (self.groups.iter())
            .map(|group| {
                let pool = (self.pools.get(group.pool))
                    .ok_or_else(|| invalid("cache group names no pool"))?;
                if pool.block == 0 || pool.bytes == 0 {
                    return Err(invalid("cache pool has empty blocks"));
                }
                // A device table holds a page's byte offset within its layer as an I32.
                let size = (blocks.checked_mul(pool.bytes))
                    .filter(|&size| size <= 1 << 31)
                    .ok_or_else(|| invalid("cache blocks pass I32 byte offsets"))?;
                Ok(iter::repeat_n(size, group.layers.len()))
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(groups.into_iter().flatten().collect())
    }
}

/// The storage of a spec, with the views one decoder keeps of it.
pub struct Cache {
    /// Each layer's pages back to back, group by group in spec order.
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
        let layers = (spec.sizes(blocks)?.into_iter())
            .map(|size| HbmBuffer::alloc(device, size).map_err(io::Error::other))
            .collect::<io::Result<_>>()?;
        Ok(Self {
            layers,
            blocks,
            spec: spec.clone(),
            views: Mutex::new(Views::Empty),
        })
    }

    /// Each layer's storage, group by group in spec order.
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

    /// One pool of `bytes`-byte pages, one paged group over `layers`.
    fn paged(bytes: usize, layers: Range<usize>) -> CacheSpec {
        CacheSpec {
            pools: vec![Pool { block: 16, bytes }],
            groups: vec![Group {
                pool: 0,
                layers,
                kind: Kind::Paged,
                window: None,
            }],
        }
    }

    /// A cache of no storage, for loans whose views read none.
    fn detached() -> Cache {
        Cache {
            layers: Vec::new(),
            blocks: 1,
            spec: paged(64, 0..0),
            views: Mutex::new(Views::Empty),
        }
    }

    /// Every layer of every group holds a page of its pool for each block, group by group.
    #[test]
    fn sizes_each_layer_a_page_a_block() -> io::Result<()> {
        let mut spec = paged(64, 0..2);
        spec.pools.push(Pool {
            block: 16,
            bytes: 32,
        });
        spec.groups.push(Group {
            pool: 1,
            layers: 2..3,
            kind: Kind::State,
            window: None,
        });
        assert_eq!(spec.sizes(3)?, [192, 192, 96]);
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
            let error = spec.sizes(blocks).expect_err(message);
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{message}");
            assert_eq!(error.to_string(), message);
        }
        assert!(paged(64, 0..1).sizes(1 << 25).is_ok());
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
