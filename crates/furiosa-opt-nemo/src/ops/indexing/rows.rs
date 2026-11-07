//! Readout input: the live rows of a call, borrowed from a row slice into a compiled envelope.

use crate::{
    DeviceModule, Module, Unary,
    compose::Count,
    ops::{
        indexing::Store,
        layout::{Dm, Live, axes::Chip, axes::Lanes, axes::Tile},
        normalization::Spread,
        vocabulary::{Greedy, LmHead, Tokens, projection::Scores},
    },
};
use furiosa_opt_std::prelude::*;
use std::{io, marker::PhantomData};

/// The rows of a slice, the live rows `0..len` of an `R`-row envelope.
#[derive(DeviceSend)]
pub struct Rows<'a, D: Scalar, W: AxisName, R: AxisName> {
    source: &'a super::RowStorage<W, D>,
    /// The source's row count, which the device reads at the launch's start.
    count: Count,
    shape: PhantomData<R>,
}

impl<'a, D: Scalar, W: AxisName, R: AxisName> Rows<'a, D, W, R> {
    /// Borrow `source` and upload its row count; refuse more rows than the envelope holds.
    pub async fn new(device: &mut Device, source: &'a super::RowStorage<W, D>) -> io::Result<Self> {
        if source.len() > R::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "rows exceed the compiled envelope",
            ));
        }
        let rows = i32::try_from(source.len()).expect("an envelope's rows fit I32");
        let count = HostTensor::<i32, m![1]>::from_vec([rows])
            .to_hbm(&mut device.pdma)
            .await
            .map_err(io::Error::other)?;
        Ok(Self {
            source,
            count,
            shape: PhantomData,
        })
    }
}

impl<D: MaterializableScalar, W: AxisName, R: AxisName> Rows<'_, D, W, R> {
    /// The live rows staged as the first rows of an envelope tensor, whose later rows are
    /// unspecified.
    pub(in crate::ops) fn stage(
        &self,
        device: &mut Device,
    ) -> (HbmTensor<D, Chip, m![R, W]>, Live<R>) {
        let live = Live::<R>::read(&self.count);
        let mut output: HbmTensor<D, Chip, m![R, W]> = HbmTensor::new();
        for row in 0..live.rows() {
            self.source[row].view().to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![R], 1, m![R = 1 #{!} R, W]>(row),
            );
        }
        (output, live)
    }
}

impl<W: AxisName, R: AxisName> Module<(&mut Device, &mut Rows<'_, bf16, W, R>)> for LmHead<W> {
    type Output = HbmTensor<f32, Chip, Scores<R>>;

    fn forward(&self, (device, input): (&mut Device, &mut Rows<'_, bf16, W, R>)) -> Self::Output {
        let (hidden, live) = input.stage(device);
        let hidden: Spread<W, R> = hidden.to_dm(&mut device.tdma);
        self.forward((device, (&hidden, live)))
    }
}

impl<W: AxisName, R: AxisName> Module<(&mut Device, &mut Rows<'_, bf16, W, R>)> for Greedy<W> {
    type Output = HbmTensor<i32, Chip, m![R]>;

    fn forward(&self, (device, input): (&mut Device, &mut Rows<'_, bf16, W, R>)) -> Self::Output {
        let (hidden, live) = input.stage(device);
        let hidden: Spread<W, R> = hidden.to_dm(&mut device.tdma);
        let tokens: Tokens<R> = self
            .forward((&mut *device, (&hidden, live)))
            .to_hbm(&mut device.tdma);
        tokens.to_hbm(&mut device.tdma)
    }
}

/// The scores `head` gives the live rows of a call in feature slices, and with a `select`, each
/// row's token from its own scores; result rows past the live count are unspecified.
#[derive(DeviceSend)]
pub struct Scored<Head, Select = ()> {
    pub head: Head,
    pub select: Select,
}

impl<'h, Head> Scored<&'h Head> {
    fn scores<W: AxisName, R: AxisName, V: AxisName>(
        &self,
        device: &mut Device,
        input: &mut Rows<'_, bf16, W, R>,
    ) -> (HbmTensor<bf16, Chip, m![R, V]>, Live<R>)
    where
        Head: for<'x> DeviceModule<
                &'x Dm<Lanes<R>, Spread<W, R>>,
                Output = Dm<Lanes<R>, Spread<V, R>>,
            >,
    {
        let (hidden, live) = input.stage(device);
        let hidden = Dm {
            value: hidden.to_dm(&mut device.tdma),
            tiles: live,
        };
        let scores = self.head.forward((&mut *device, &hidden));
        (Store.forward((device, &scores)), live)
    }
}

impl<'h, W: AxisName, R: AxisName, V: AxisName, Head>
    Module<(&mut Device, &mut Rows<'_, bf16, W, R>)> for Scored<&'h Head>
where
    Head: for<'x> DeviceModule<&'x Dm<Lanes<R>, Spread<W, R>>, Output = Dm<Lanes<R>, Spread<V, R>>>,
{
    type Output = HbmTensor<bf16, Chip, m![R, V]>;

    fn forward(&self, (device, input): (&mut Device, &mut Rows<'_, bf16, W, R>)) -> Self::Output {
        self.scores(device, input).0
    }
}

impl<'h, 's, W: AxisName, R: AxisName, V: AxisName, Head, Select>
    Module<(&mut Device, &mut Rows<'_, bf16, W, R>)> for Scored<&'h Head, &'s Select>
where
    Head: for<'x> DeviceModule<&'x Dm<Lanes<R>, Spread<W, R>>, Output = Dm<Lanes<R>, Spread<V, R>>>,
    Select: for<'x> DeviceModule<
            &'x HbmTensor<bf16, Chip, m![V]>,
            Output = HbmTensor<i32, Chip, m![1 # 4]>,
        >,
{
    type Output = HbmTensor<i32, Chip, m![R]>;

    fn forward(&self, (device, input): (&mut Device, &mut Rows<'_, bf16, W, R>)) -> Self::Output {
        let (scores, live) = Scored {
            head: self.head,
            select: (),
        }
        .scores(device, input);
        let mut tokens: Tokens<R> = HbmTensor::new();
        for row in 0..live.rows() {
            let mut logits: HbmTensor<bf16, Chip, m![V]> = HbmTensor::new();
            scores
                .view()
                .tile::<m![R], 1, m![R = 1 # R, V]>(row)
                .to_hbm_view(&mut device.tdma, logits.view_mut());
            self.select
                .forward((&mut *device, &logits))
                .view()
                .to_hbm_view(
                    &mut device.tdma,
                    tokens
                        .view_mut()
                        .tile::<m![R], 1, m![R = 1 #{!} R, 1 # 4]>(row),
                );
        }
        tokens.to_hbm(&mut device.tdma)
    }
}

/// A function of each row, applied to the live rows of a call one tile of up to sixteen rows a
/// trip, for a call of at least one row; result rows past the live count are unspecified.
#[derive(DeviceSend)]
pub struct Rowwise<F>(pub F);

type RowTile<R> = Symbol<Tile<R, 16>>;

impl<F, D, W, R, S> Module<(&mut Device, &mut Rows<'_, D, W, R>)> for Rowwise<F>
where
    D: MaterializableScalar,
    W: AxisName,
    R: AxisName,
    S: AxisName,
    F: Unary<
            HbmTensor<D, Chip, m![{ RowTile<R> }, W]>,
            Output = HbmTensor<f32, Chip, m![{ RowTile<R> }, S]>,
        >,
{
    type Output = HbmTensor<f32, Chip, m![R, S]>;

    fn forward(&self, (device, input): (&mut Device, &mut Rows<'_, D, W, R>)) -> Self::Output {
        let live = Live::<R>::read(&input.count);
        let tiles = live.retile::<Tile<R, 16>>().count();
        let mut output: HbmTensor<f32, Chip, m![R, S]> = HbmTensor::new();
        let mut apply = |device: &mut Device, tile: usize| {
            let mut rows: HbmTensor<D, Chip, m![{ RowTile<R> }, W]> = HbmTensor::new();
            #[unroll]
            for lane in 0..<RowTile<R> as M>::SIZE {
                let row = tile * <RowTile<R> as M>::SIZE + lane;
                if row < live.rows() {
                    input.source[row].view().to_hbm_view(
                        &mut device.tdma,
                        rows.view_mut().tile::<RowTile<R>, 1, m![
                            { RowTile<R> } = 1 #{!} ({ RowTile<R> }), W
                        ]>(lane),
                    );
                }
            }
            let result = self.0.forward((&mut *device, &rows));
            // SAFETY: the destination's row tile supplies the tile's outer offset.
            let result = unsafe { result.view().reshape::<Chip, m![R % ({ RowTile<R> }), S]>() };
            result.to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![R / ({ RowTile<R> })], 1, m![
                    R / ({ RowTile<R> }) = 1 #{!} (R / ({ RowTile<R> })),
                    R % ({ RowTile<R> }),
                    S
                ]>(tile),
            );
        };
        // A launch has a live row, so its first tile runs outside any runtime condition, where a
        // not yet implemented function's panic still lowers.
        apply(device, 0);
        for trip in 0..const { <m![R / ({ RowTile<R> })] as M>::SIZE - 1 } {
            if trip + 1 < tiles {
                apply(device, trip + 1);
            }
        }
        output
    }
}

/// Copy a slice's rows into the first rows of a compiled envelope; later rows are unspecified.
#[derive(DeviceSend)]
pub struct Copy<Width, Rows, D = bf16>(PhantomData<(Width, Rows, D)>);

impl<W, R, D> Default for Copy<W, R, D> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<W: AxisName, R: AxisName, D: MaterializableScalar>
    Module<(&mut Device, &super::RowStorage<W, D>)> for Copy<W, R, D>
{
    type Output = HbmTensor<D, Chip, m![R, W]>;

    fn forward(&self, (device, source): (&mut Device, &super::RowStorage<W, D>)) -> Self::Output {
        let mut output: HbmTensor<D, Chip, m![R, W]> = HbmTensor::new();
        #[unroll]
        for row in 0..R::SIZE {
            if row < source.len() {
                source[row].view().to_hbm_view(
                    &mut device.tdma,
                    output.view_mut().tile::<m![R], 1, m![R = 1 #{!} R, W]>(row),
                );
            }
        }
        output
    }
}

impl<W: AxisName, R: AxisName, D: MaterializableScalar>
    Module<(&mut Device, &mut &super::RowStorage<W, D>)> for Copy<W, R, D>
{
    type Output = HbmTensor<D, Chip, m![R, W]>;

    fn forward(
        &self,
        (device, source): (&mut Device, &mut &super::RowStorage<W, D>),
    ) -> Self::Output {
        self.forward((device, *source))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Then;
    use crate::ops::linear::Affine;
    use crate::ops::vocabulary::Argmax;
    use furiosa_opt_std::runtime::launch;

    type Envelope = crate::ops::layout::axes::Rows<8>;
    axes![Width = 2048, Scores = 4096];
    type Head = Affine<HbmTensor<bf16, Chip, m![Scores, Width]>, ()>;

    #[device(chip = 1)]
    fn tokens(
        device: &mut Device,
        head: &Scored<&'static Head, &'static Argmax<Scores>>,
        input: &mut Rows<'_, bf16, Width, Envelope>,
    ) -> HbmTensor<i32, Chip, m![Envelope]> {
        head.forward((device, input))
    }

    #[device(chip = 1)]
    fn token(
        device: &mut Device,
        (head, select): (&Head, &Argmax<Scores>),
        input: &HbmTensor<bf16, Chip, m![Width]>,
    ) -> HbmTensor<i32, Chip, m![1 # 4]> {
        Then(head, select).forward((device, input))
    }

    fn values(count: usize, mut state: u64) -> Vec<bf16> {
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bf16::from_f32((state >> 40) as f32 / (1u64 << 21) as f32 - 4.0)
            })
            .collect()
    }

    async fn upload<E: M>(device: &mut Device, values: &[bf16]) -> HbmTensor<bf16, Chip, E> {
        HostTensor::<bf16, E>::from_vec(values.to_vec())
            .to_hbm(&mut device.pdma)
            .await
            .unwrap()
    }

    /// One launch selects each live row's token as the head and selector select it for the row
    /// alone.
    #[tokio::test(flavor = "current_thread")]
    async fn selects_rows_as_alone() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(tokens.topology()).unwrap();
        let head: &'static Head = Box::leak(Box::new(Affine::new(
            upload::<m![Scores, Width]>(&mut device, &values(4096 * 2048, 7)).await,
            (),
        )));
        let select: &'static Argmax<Scores> = Box::leak(Box::new(Argmax {
            indices: HostTensor::<i32, m![Scores]>::from_vec((0..4096).map(|i| 4095 - i))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        }));
        let inputs = values(8 * 2048, 11);
        let mut rows = Vec::new();
        for row in 0..8 {
            rows.push(upload::<m![Width]>(&mut device, &inputs[row * 2048..][..2048]).await);
        }
        for live in [8, 3] {
            // SAFETY: the views only read rows that outlive the launch.
            let source: Slice<HbmTensor<bf16, Chip, m![Width]>> = rows[..live]
                .iter()
                .map(|row| unsafe { row.view().alias() })
                .collect();
            let mut input = Rows::<bf16, Width, Envelope>::new(&mut device, &source)
                .await
                .unwrap();
            let selected = launch(tokens, (&mut device, &Scored { head, select }, &mut input))
                .await
                .unwrap()
                .to_host::<m![Envelope]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (row, input) in rows[..live].iter().enumerate() {
                let alone = launch(token, (&mut device, (head, select), input))
                    .await
                    .unwrap()
                    .to_host::<m![1 # 4]>(&mut device.pdma)
                    .await
                    .unwrap()
                    .into_vec()[0];
                assert_eq!(selected[row], alone, "row {row} of {live}");
            }
        }
    }
}
