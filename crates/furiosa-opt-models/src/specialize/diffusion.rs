//! The run of a packed text, audio or video row's features that each slice of a strip holds.

#![deny(clippy::arbitrary_source_item_ordering)]

use core::marker::PhantomData;

use furiosa_opt_std::prelude::*;

/// Features of a row one slice of a 64-row strip holds: at most six whole 256-byte bursts over at
/// most eight slices, so a span's two FP32 operands (6 KiB) and a pass's cache fit 8 KiB.
#[derive(Clone, Debug)]
pub(crate) struct Run<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Run<W> {
    const NAME: Ident = Ident::new("Run");
    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(128), "rows come in 256-byte bursts");
        let bursts = W::SIZE / 128;
        let mut each = 6;
        while bursts % each != 0 || bursts / each > 8 {
            each -= 1;
            assert!(each > 0, "no run fits the slices of a row");
        }
        128 * each
    };
}

/// Features of a row one slice of a 16-row strip holds: a row spreads over up to 32 slices.
#[derive(Clone, Debug)]
pub(crate) struct Spread<W>(PhantomData<W>);

impl<W: AxisName> AxisName for Spread<W> {
    const NAME: Ident = Ident::new("Spread");
    const SIZE: usize = {
        assert!(W::SIZE.is_multiple_of(128), "rows come in 256-byte bursts");
        let bursts = W::SIZE / 128;
        let mut each = 1;
        while bursts % each != 0 || bursts / each > 32 {
            each += 1;
            assert!(each <= 6, "no run fits the slices of a row");
        }
        128 * each
    };
}
