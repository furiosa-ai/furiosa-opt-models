use crate::ops::layout::axes::{Parallel, Parallelism};
use crate::ops::linear::Affine;
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData};

pub trait Layout: Clone + std::fmt::Debug + 'static {
    type Hidden: AxisName;
    type Vocabulary: AxisName;

    fn token(rank: usize, entry: usize) -> usize;
}

#[derive(Clone, Debug)]
pub struct Local<L>(PhantomData<L>);

impl<L: Layout> AxisName for Local<L> {
    const NAME: Ident = Ident::new("Vocabulary");
    const SIZE: usize = L::Vocabulary::SIZE / 2;
}

pub type Chips = <Parallel<2> as Parallelism>::Chip;
type Hidden<L> = Symbol<<L as Layout>::Hidden>;
pub type Stored<L> = m![{ Hidden<L> } / 256, { Symbol<Local<L>> }, { Hidden<L> } % 256];
pub type Packed<L> = Affine<Parameters<L>, [HbmTensor<bf16, Chips, m![{ Symbol<Local<L>> }]>; 0]>;

#[derive(DeviceSend)]
pub struct Parameters<L: Layout> {
    pub weight: HbmTensor<bf16, Chips, Stored<L>>,
}

#[derive(DeviceSend)]
pub struct Sampling<L: Layout, C: M = Chips> {
    pub stops: HbmTensor<f32, C, m![{ Symbol<Local<L>> }]>,
}

/// One checked permutation supplies coefficient rows, reversed IDs and stop entries at Load.
pub struct Order<L: Layout> {
    tokens: Vec<usize>,
    stops: Vec<usize>,
    layout: PhantomData<L>,
}

impl<L: Layout> Order<L> {
    pub fn new(stops: &[usize]) -> io::Result<Self> {
        const {
            assert!(L::Hidden::SIZE % 256 == 0);
            assert!(L::Vocabulary::SIZE > 0 && L::Vocabulary::SIZE % 2 == 0);
        }
        let count = L::Vocabulary::SIZE;
        i32::try_from(count - 1).map_err(io::Error::other)?;
        if stops.iter().any(|&token| token >= count) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "stop ID exceeds vocabulary",
            ));
        }
        let mut seen = vec![false; count];
        let tokens = (0..2)
            .flat_map(|rank| (0..Local::<L>::SIZE).map(move |entry| L::token(rank, entry)))
            .map(|token| {
                if token >= count || seen[token] {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "vocabulary order is not a permutation",
                    ));
                }
                seen[token] = true;
                Ok(token)
            })
            .collect::<io::Result<Vec<_>>>()?;
        let mut stops = stops.to_vec();
        stops.sort_unstable();
        stops.dedup();
        Ok(Self {
            tokens,
            stops,
            layout: PhantomData,
        })
    }

    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    pub fn stops(&self) -> &[usize] {
        &self.stops
    }
}
