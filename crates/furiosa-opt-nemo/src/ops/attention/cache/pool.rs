use super::{Blocks, Layer};
use crate::compose::{
    Space,
    step::{Lend, Prepared},
};
use crate::ops::layout::axes::Chip;
use crate::{Contexts, Module};
use furiosa_opt_std::prelude::*;
use std::io;

/// Fused block storage for `N` equal-sized layer ranges.
/// Layer `i` begins at block offset `i * entries()` within the fused blocks.
#[derive(DeviceSend)]
pub struct Pool<
    Heads: AxisName,
    Dimension: AxisName,
    D: Scalar = bf16,
    Chips: M = Chip,
    const N: usize = 1,
> {
    /// Block handles concatenated in source-layer order.
    pub(crate) blocks: Blocks<Heads, Dimension, D, Chips>,
}

/// The blocks of exactly `N` nonempty layers with equal block counts, in layer order; the layers
/// are consumed.
pub(crate) fn fuse<D: MaterializableScalar, Chips: M, E: M, const N: usize>(
    layers: impl IntoIterator<Item = Slice<HbmTensor<D, Chips, E>>>,
) -> io::Result<Slice<HbmTensor<D, Chips, E>>> {
    let layers: Vec<_> = layers.into_iter().collect();
    let entries = layers.first().map_or(0, |layer| layer.len());
    if layers.len() != N
        || entries == 0
        || entries.checked_mul(N).is_none()
        || layers.iter().any(|layer| layer.len() != entries)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "layers must have equal nonempty block ranges",
        ));
    }
    Ok(layers
        .iter()
        .flat_map(|layer| {
            layer
                .iter()
                // SAFETY: The source layers drop on return, so the aliases become their only
                // handles.
                .map(|block| unsafe { block.view().alias() })
        })
        .collect())
}

impl<H: AxisName, W: AxisName, D: Scalar, Chips: M, const N: usize> Pool<H, W, D, Chips, N> {
    /// Fuse exactly `N` nonempty layers with equal block counts, consuming them.
    pub fn new(layers: impl IntoIterator<Item = Blocks<H, W, D, Chips>>) -> io::Result<Self>
    where
        D: MaterializableScalar,
    {
        let tokens = fuse::<_, _, _, N>(layers.into_iter().map(|layer| layer.tokens))?;
        Ok(Self {
            blocks: Blocks { tokens },
        })
    }

    /// Number of blocks in each source layer's range.
    pub fn entries(&self) -> usize {
        self.blocks.tokens.len() / N
    }

    /// The fused blocks of every layer.
    pub fn root(&mut self) -> &mut Blocks<H, W, D, Chips> {
        &mut self.blocks
    }

    /// The range of layer `index`.
    pub fn layer(&mut self, index: usize) -> Layer<'_, H, W, D, Chips> {
        let entry = index * self.entries();
        Layer {
            blocks: &mut self.blocks,
            entry,
        }
    }
}

/// Each layer reads its own range of the pool and the shared context.
impl<'p, L, D, X, P, H: AxisName, W: AxisName, S: Scalar, Chip: M, const N: usize> Contexts<L, D, X>
    for (&mut Pool<H, W, S, Chip, N>, &'p P)
where
    L: for<'d, 'x, 'c, 'r> Module<
            (&'d mut D, (&'x X, &'c mut Layer<'r, H, W, S, Chip>, &'p P)),
            Output = X,
        >,
{
    fn apply(&mut self, layer: &L, device: &mut D, hidden: X, index: usize) -> X {
        let mut range = self.0.layer(index);
        layer.forward((device, (&hidden, &mut range, self.1)))
    }
}

/// A pool's layers read the context the call's step prepared; the step's other cache operands are
/// its preparation's alone.
impl<'c, 'p, R, T: Space, P, H: AxisName, W: AxisName, S: Scalar, Chip: M, const N: usize>
    Lend<&'p Prepared<T, P>> for &'c mut (&mut Pool<H, W, S, Chip, N>, R)
{
    type Context = (&'c mut Pool<H, W, S, Chip, N>, &'p P);

    fn lend(self, prepared: &'p Prepared<T, P>) -> Self::Context {
        (&mut *self.0, prepared.context())
    }
}

/// Each of two pools' layers reads the context its own geometry prepared.
impl<'c, 'p, R, T: Space, P, Q, H, W, J, E, S: Scalar, Chip: M, const N: usize, const K: usize>
    Lend<&'p Prepared<T, (P, Q)>>
    for &'c mut (
        (&mut Pool<H, W, S, Chip, N>, &mut Pool<J, E, S, Chip, K>),
        R,
    )
where
    H: AxisName,
    W: AxisName,
    J: AxisName,
    E: AxisName,
{
    type Context = (
        (&'c mut Pool<H, W, S, Chip, N>, &'p P),
        (&'c mut Pool<J, E, S, Chip, K>, &'p Q),
    );

    fn lend(self, prepared: &'p Prepared<T, (P, Q)>) -> Self::Context {
        let ((first, second), _) = self;
        let (p, q) = prepared.context();
        ((&mut **first, p), (&mut **second, q))
    }
}

/// Each layer reads its own range of the pool and its own context.
impl<L, D, X, P, H: AxisName, W: AxisName, S: Scalar, Chip: M, const N: usize> Contexts<L, D, X>
    for &mut (&mut Pool<H, W, S, Chip, N>, [P; N])
where
    L: for<'d, 'x, 'c, 'v, 'r> Module<
            (&'d mut D, (&'x X, &'c mut Layer<'r, H, W, S, Chip>, &'v P)),
            Output = X,
        >,
{
    fn apply(&mut self, layer: &L, device: &mut D, hidden: X, index: usize) -> X {
        let (pool, contexts) = &mut **self;
        let mut range = pool.layer(index);
        layer.forward((device, (&hidden, &mut range, &contexts[index])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::attention::tests::{Head, KvHeads};
    use crate::{Branch, Interleave, Layers};

    /// Layers with unequal block counts do not fuse.
    #[test]
    fn rejects_unequal_layer_ranges() {
        let _lease = crate::tests::DEVICE.blocking_lock();
        let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let layers: [Blocks<KvHeads, Head>; 2] = [1, 2].map(|entries| Blocks {
            tokens: (0..entries)
                .map(|_| HbmTensor::alloc(&device).unwrap())
                .collect(),
        });
        let result = Pool::<KvHeads, Head, bf16, Chip, 2>::new(layers);
        assert_eq!(
            result.err().map(|error| error.kind()),
            Some(io::ErrorKind::InvalidInput)
        );
    }

    struct Record(usize);

    impl
        Module<(
            &mut Vec<(usize, usize)>,
            (&usize, &mut Layer<'_, KvHeads, Head>, &()),
        )> for Record
    {
        type Output = usize;

        fn forward(
            &self,
            (trace, (hidden, pool, _)): (
                &mut Vec<(usize, usize)>,
                (&usize, &mut Layer<'_, KvHeads, Head>, &()),
            ),
        ) -> usize {
            trace.push((self.0, pool.entry));
            hidden * 2 + self.0
        }
    }

    /// Layer parameters and pool ranges remain paired in execution order.
    #[test]
    fn preserves_layer_order() {
        let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mut cache = Pool::<KvHeads, Head, bf16, Chip, 3> {
            blocks: Blocks {
                tokens: (0..12)
                    .map(|_| HbmTensor::alloc(&device).unwrap())
                    .collect(),
            },
        };
        let mut trace = Vec::new();
        let model = Layers {
            layers: [Record(1), Record(2), Record(3)],
        };
        let hidden = model.forward((&mut trace, (0, (&mut cache, &()))));
        assert_eq!((hidden, trace), (11, vec![(1, 0), (2, 4), (3, 8)]));
    }

    /// Repeated heterogeneous periods preserve weights, independent pool strides and ordering.
    #[test]
    fn preserves_interleaved_order() {
        const ORDER: &[Branch] = &[
            Branch::First,
            Branch::First,
            Branch::Second,
            Branch::First,
            Branch::First,
            Branch::Second,
        ];
        let device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let model = Interleave::<_, _, ORDER> {
            first: [Record(1), Record(2), Record(4), Record(5)],
            second: [Record(3), Record(6)],
        };
        let mut first = Pool::<KvHeads, Head, bf16, Chip, 4> {
            blocks: Blocks {
                tokens: (0..16)
                    .map(|_| HbmTensor::alloc(&device).unwrap())
                    .collect(),
            },
        };
        let mut second = Pool::<KvHeads, Head, bf16, Chip, 2> {
            blocks: Blocks {
                tokens: (0..16)
                    .map(|_| HbmTensor::alloc(&device).unwrap())
                    .collect(),
            },
        };
        let mut trace = Vec::new();
        let hidden = model.forward((&mut trace, (0, ((&mut first, &()), (&mut second, &())))));
        assert_eq!(
            (hidden, trace),
            (120, vec![(1, 0), (2, 4), (3, 0), (4, 8), (5, 12), (6, 8)])
        );
    }
}
