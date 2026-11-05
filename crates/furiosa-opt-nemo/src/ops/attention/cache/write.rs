use super::Scatter;
use crate::{Module, compose::Space, ops::layout::Dm};
use furiosa_opt_std::prelude::*;

/// Physical page and token coordinates of a cache-write iteration space.
pub trait Mapping: Space {
    type Page: M;
    type Payload: M;
}

/// Publish an already selected resident tile at its rows' byte offsets in a layer's pages.
impl<T: Mapping, D: MaterializableScalar, B: M, C: M, S: M, E: M, R: M>
    Module<(
        &mut Device,
        (
            &mut Slice<HbmTensor<D, B, m![{ T::Page }, { T::Payload }]>>,
            (usize, &HbmTensor<i32, B, R>),
            Dm<T, DmTensorView<'_, D, B, C, S, E>>,
        ),
    )> for Scatter
{
    type Output = ();

    fn forward(
        &self,
        (device, (pages, (entry, offsets), rows)): (
            &mut Device,
            (
                &mut Slice<HbmTensor<D, B, m![{ T::Page }, { T::Payload }]>>,
                (usize, &HbmTensor<i32, B, R>),
                Dm<T, DmTensorView<'_, D, B, C, S, E>>,
            ),
        ),
    ) {
        pages
            .scatter::<T::Page, T::Payload>()
            .entry(entry)
            .by_byte_offsets(offsets.view())
            .from_dm_view(&mut device.tdma, rows.value);
    }
}
