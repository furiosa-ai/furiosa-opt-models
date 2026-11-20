//! Shared text-envelope preparation for a caller-selected prepared attention operation.

use super::{Copy, Rows, Text};
use crate::Op;
use crate::ops::attention::flash::{self, HeadPairs, Rows as TileRows};
use crate::ops::index::packed::Span;
use crate::tensor::scalar;
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;

type Inputs<'a, const LIVE: usize, const ROWS: usize, W, K, Chips> = (
    &'a Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
    &'a Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
    &'a Rows<bf16, Text<LIVE, ROWS>, W, Chips>,
    (&'a flash::Visible<K, Chips>, &'a scalar::Value<Chips>),
);

/// Repeat finite envelope rows into the query tile and head-major key spans, attend with
/// scattered visibility, then extract the original envelope. Every input row, including
/// semantically absent padding, must be initialized and finite. `live` is the key iteration
/// bound, independently of how many keys the visibility marks present.
pub(super) fn attend<
    const LIVE: usize,
    const ROWS: usize,
    W: AxisName,
    Heads: AxisName,
    K: AxisName,
    Chips: M,
    A,
>(
    device: &mut Device,
    attention: &A,
    (query, key, value, (visible, live)): Inputs<'_, LIVE, ROWS, W, K, Chips>,
) -> Rows<bf16, Text<LIVE, ROWS>, W, Chips>
where
    A: for<'a> Op<
            (
                &'a flash::Queries<TileRows, HeadPairs<W>, Chips>,
                &'a flash::Tokens<K, HeadPairs<W>, Chips>,
                &'a flash::Tokens<K, HeadPairs<W>, Chips>,
                (&'a flash::Scattered<'a, K, Chips>, &'a scalar::Value<Chips>),
            ),
            Output = flash::Queries<TileRows, HeadPairs<W>, Chips>,
        >,
{
    const {
        assert!(
            Heads::SIZE == 2 * HeadPairs::<W>::SIZE,
            "a chip attends its heads in pairs"
        );
        assert!(TileRows::SIZE.is_multiple_of(ROWS));
        assert!((K::SIZE * flash::Span::SIZE).is_multiple_of(ROWS));
    };
    let mut tile: Rows<bf16, TileRows, W, Chips> = HbmTensor::new();
    for copy in 0..TileRows::SIZE / ROWS {
        Span::<LIVE, ROWS>::new().forward(&mut *device, (query, (&mut tile, copy * ROWS)));
    }
    // SAFETY: a row is its heads in order, two a pair.
    let tile: flash::Queries<TileRows, HeadPairs<W>, Chips> = unsafe { tile.reshape() };
    let tokens = |device: &mut Device, rows: &Rows<bf16, Text<LIVE, ROWS>, W, Chips>| {
        let mut tokens = HbmTensor::new();
        for copy in 0..K::SIZE * flash::Span::SIZE / ROWS {
            Copy::<K, ROWS>(PhantomData).forward(device, ((rows, 0), (&mut tokens, copy * ROWS)));
        }
        tokens
    };
    let (keys, values) = (tokens(device, key), tokens(device, value));
    let output = Op::forward(
        attention,
        &mut *device,
        (&tile, &keys, &values, (&flash::Scattered(visible), live)),
    );
    // SAFETY: a row is its heads in order, two a pair.
    let output: Rows<bf16, TileRows, W, Chips> = unsafe { output.reshape() };
    Span::<LIVE, ROWS>::new().forward(device, (&output, 0))
}
