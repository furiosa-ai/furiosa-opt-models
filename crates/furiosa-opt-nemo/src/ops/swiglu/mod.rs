//! Packed gate/up projections compute down(silu(gate(x)) * up(x)) on BF16 rows.

use furiosa_opt_std::prelude::*;

use crate::ops::activation::SwiGlu;
use crate::ops::layout::{Dm, axes::Lanes};
use crate::ops::normalization::Spread;
use crate::{Module, ops::layout::Live, ops::linear::Groups, ops::linear::Linear};

use crate::ops::layout::axes::Chip;
use crate::ops::vector::Zip;

mod giant;
mod medium;
mod paired;
mod resident;
mod rows;
pub mod spans;
mod tiles;

/// Intermediate feature axis shared by dense feed-forward implementations.
#[derive(Clone, Debug)]
pub struct Intermediate<const N: usize, const PADDED: usize = N>;

impl<const N: usize, const PADDED: usize> AxisName for Intermediate<N, PADDED> {
    const NAME: Ident = Ident::new("Intermediate");
    const SIZE: usize = N;
}

/// The two weight matrices packed as gate followed by up.
#[derive(Debug, Clone)]
pub struct GateUp<I>(std::marker::PhantomData<I>);

impl<I: AxisName> AxisName for GateUp<I> {
    const NAME: Ident = Ident::new("GateUp");
    const SIZE: usize = I::SIZE * 2;
}

/// Storage width of the down projection, including its kernel's zero padding.
#[derive(Debug, Clone)]
pub struct Down<I>(std::marker::PhantomData<I>);

impl<const N: usize, const PADDED: usize> AxisName for Down<Intermediate<N, PADDED>> {
    const NAME: Ident = Intermediate::<N, PADDED>::NAME;
    const SIZE: usize = {
        assert!(PADDED >= N);
        PADDED
    };
}

/// SwiGLU with a packed gate/up projection and a down projection.
#[derive(DeviceSend)]
pub struct Mlp<
    Hidden: AxisName,
    Intermediate: AxisName,
    Fused = Linear<Hidden, GateUp<Intermediate>>,
    Projection = Linear<Down<Intermediate>, Hidden>,
> where
    Down<Intermediate>: AxisName,
{
    gate_up: Fused,
    down: Projection,
    features: core::marker::PhantomData<(Hidden, Intermediate)>,
}

impl<Hidden: AxisName, Intermediate: AxisName, Fused, Projection>
    Mlp<Hidden, Intermediate, Fused, Projection>
where
    Down<Intermediate>: AxisName,
{
    pub const fn new(gate_up: Fused, down: Projection) -> Self {
        Self {
            gate_up,
            down,
            features: core::marker::PhantomData,
        }
    }
}

// Gate then up in the packed projection.
axes![Term = 2];

/// One row group of an envelope of `Rows`, the envelope a live group runs alone as.
type Group<Rows> = crate::ops::layout::axes::Tile<Rows, 8>;

impl<Hidden: AxisName, Intermediate: AxisName> Mlp<Hidden, Intermediate>
where
    Down<Intermediate>: AxisName,
{
    /// `compute` of each live row group alone, run as an envelope of that one group; later rows
    /// land zeros. A group loads the weights its envelope loads, so this serves envelopes of at
    /// most two groups, where a step with a group past its live rows has one live group.
    fn each_group<Rows: AxisName>(
        &self,
        device: &mut Device,
        hidden: &Spread<Hidden, Rows>,
        live: Live<Rows>,
        compute: impl Fn(
            &Self,
            &mut Device,
            &Spread<Hidden, Group<Rows>>,
        ) -> Spread<Hidden, Group<Rows>>,
    ) -> Spread<Hidden, Rows> {
        type Chunk<Hidden> = crate::ops::normalization::Chunk<Hidden>;
        type Slices<Hidden> = crate::ops::normalization::FeatureSlices<Hidden>;
        type Cluster = crate::ops::normalization::Cluster;
        let mut output: Spread<Hidden, Rows> = DmTensor::new();
        output
            .view_mut()
            .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
        for group in 0..live.count() {
            let selected = hidden
                .view()
                .tile::<m![Rows / ({ Symbol<Group<Rows>> })], 1, m![
                    Rows / ({ Symbol<Group<Rows>> }) = 1 # (Rows / ({ Symbol<Group<Rows>> })),
                    Rows % ({ Symbol<Group<Rows>> }),
                    Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>(group);
            // SAFETY: the group's rows keep their cells and strides; only the row axis is renamed.
            let selected = unsafe {
                selected.reshape::<Chip, Cluster, Slices<Hidden>, m![
                    { Symbol<Group<Rows>> } # Rows, Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>()
            }
            .unpad::<m![{ Symbol<Group<Rows>> }, Hidden % ({ Symbol<Chunk<Hidden>> })]>();
            let mut rows: Spread<Hidden, Group<Rows>> = DmTensor::new();
            selected.to_dm_view(&mut device.tdma, rows.view_mut());
            let result = compute(self, device, &rows);
            // SAFETY: the destination's row tile supplies the group's outer offset.
            let result = unsafe {
                result.view().reshape::<Chip, Cluster, Slices<Hidden>, m![
                    Rows % ({ Symbol<Group<Rows>> }), Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>()
            };
            result.to_dm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Rows / ({ Symbol<Group<Rows>> })], 1, m![
                    Rows / ({ Symbol<Group<Rows>> }) = 1 #{!} (Rows / ({ Symbol<Group<Rows>> })),
                    Rows % ({ Symbol<Group<Rows>> }),
                    Hidden % ({ Symbol<Chunk<Hidden>> })
                ]>(group),
            );
        }
        output
    }
}

/// A launch's live rows through the whole MLP, at the catalog's MLP position.
impl<'d, 'x, H: AxisName, I: AxisName, R: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<R>, Spread<H, R>>)> for Mlp<H, I>
where
    Down<I>: AxisName,
    Self: for<'e, 'y> Module<(&'e mut Device, (&'y Spread<H, R>, Live<R>)), Output = Spread<H, R>>,
{
    type Output = Dm<Lanes<R>, Spread<H, R>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<R>, Spread<H, R>>),
    ) -> Self::Output {
        Dm {
            value: self.forward((device, (&input.value, input.tiles))),
            tiles: input.tiles,
        }
    }
}

/// Each row's gate half and up half land in feature slices of their own; the SiLU of the gate
/// rounds to BF16 before it multiplies the up half, as a separate activation after a packed
/// projection rounds.
impl<'d, 'x, I: AxisName, Rows: AxisName>
    Module<(&'d mut Device, &'x Dm<Lanes<Rows>, Spread<GateUp<I>, Rows>>)> for SwiGlu
{
    type Output = Dm<Lanes<Rows>, Spread<I, Rows>>;

    fn forward(
        &self,
        (device, input): (&'d mut Device, &'x Dm<Lanes<Rows>, Spread<GateUp<I>, Rows>>),
    ) -> Self::Output {
        let packed: Packed<I, Rows> = input.value.to_hbm(&mut device.tdma);
        Dm {
            value: self.forward((device, (packed, input.tiles))),
            tiles: input.tiles,
        }
    }
}

/// Rows of the packed gate/up pair in HBM, each row's gate features before its up features.
pub(crate) type Packed<I, Rows> = HbmTensor<bf16, Chip, m![Rows, { Symbol<GateUp<I>> }]>;

/// As the feature-slice form, from the pair's rows in HBM.
impl<'d, I: AxisName, Rows: AxisName> Module<(&'d mut Device, (Packed<I, Rows>, Live<Rows>))>
    for SwiGlu
{
    type Output = Spread<I, Rows>;

    fn forward(
        &self,
        (device, (packed, _)): (&'d mut Device, (Packed<I, Rows>, Live<Rows>)),
    ) -> Self::Output {
        type C<I> = Symbol<crate::ops::normalization::Chunk<I>>;
        // SAFETY: a packed row is its gate features followed by its up features.
        let packed: HbmTensor<bf16, Chip, m![Rows, Term, I]> = unsafe { packed.reshape() };
        let half = |device: &mut Device, term: usize| {
            let mut half: Spread<I, Rows> = DmTensor::new();
            packed
                .view()
                .tile::<m![Term], 1, m![Rows, Term = 1 # 2, I]>(term)
                .to_dm_view(&mut device.tdma, half.view_mut());
            half
        };
        let (gate, up) = (half(device, 0), half(device, 1));
        let activated: Spread<I, Rows> = device
            .main
            .begin(gate.view())
            .fetch::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![Rows, I % ({ C<I> }) / 8, I % 8 / 4 % 2], m![I % 4]>()
            .vector_stash()
            .vector_fp_unary(FpUnaryOp::Sigmoid)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            .vector_widen_concat::<m![Rows, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit();
        let product: Spread<I, Rows> = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(activated.view(), up.view())
            .fetch::<m![Rows, Term, I % ({ C<I> }) / 8], m![I % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows, Term, I % ({ C<I> }) / 8], m![I % 8]>()
            .vector_init()
            .vector_zip::<
                Term,
                m![Rows, I % ({ C<I> }) / 8],
                m![Rows, I % ({ C<I> }) / 4],
                m![I % 4],
            >(FpBinaryOp::MulF(FpMulAlu::Mul0))
            .vector_final()
            .cast::<bf16, m![I % 8 # 16]>()
            .commit_trim::<m![I % 8]>()
            .commit();
        product
    }
}

/// Rows in HBM, a tile of rows at a time; as for rows on chip, the SiLU of the gate rounds to
/// BF16 before it multiplies the up half.
impl<'d, 'x, I: AxisName, R: AxisName, Chips: M>
    Module<(
        &'d mut Device,
        &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
    )> for SwiGlu
{
    type Output = HbmTensor<bf16, Chips, m![R, I]>;

    fn forward(
        &self,
        (device, input): (
            &'d mut Device,
            &'x HbmTensor<bf16, Chips, m![R, { Symbol<GateUp<I>> }]>,
        ),
    ) -> HbmTensor<bf16, Chips, m![R, I]> {
        let mut output = HbmTensor::new();
        SwiGlu.forward((device, (input, &mut output)));
        output
    }
}

#[cfg(test)]
mod tests {
    struct Fixture;
    use furiosa_opt_std::runtime::launch;

    use super::*;
    use crate::compose::{Count, tests::count};
    use crate::ops::attention::geometry::Hidden;
    use crate::ops::layout::axes::{Lanes, Rows, Sixteen};
    use crate::ops::weight::Weight;

    /// Each down row reads one channel in each of this many equal ranges of the intermediate width.
    const PARTS: usize = 8;

    fn coefficient(term: usize, row: usize, column: usize) -> bf16 {
        bf16::from_f32(((row * 31 + column * 13 + term * 47) % 37) as f32 / 16.0 - 1.125)
    }

    /// The hidden columns every gate and up row reads.
    fn columns(hidden: usize) -> [usize; 6] {
        [
            0,
            129,
            hidden / 4 - 1,
            hidden / 2,
            3 * hidden / 4 + 1,
            hidden - 1,
        ]
    }

    fn channel(intermediate: usize, row: usize, part: usize) -> usize {
        let range = intermediate / PARTS;
        part * range + (row * 17 + part * 131) % range
    }

    fn inputs(rows: usize, hidden: usize) -> Vec<bf16> {
        (0..rows)
            .flat_map(|row| {
                (0..hidden).map(move |column| {
                    bf16::from_f32(
                        ((row * 19 + column * 29 + row * column % 17) % 113) as f32 / 64.0 - 0.875,
                    )
                })
            })
            .collect()
    }

    /// down(silu(gate x) * up x) of every row with BF16 activations and FP32 sums, each with one
    /// BF16 ulp of every summed term: a device rounding one activation the other way lands there.
    fn oracle(inputs: &[bf16], hidden: usize, intermediate: usize) -> Vec<(f32, f32)> {
        inputs
            .chunks_exact(hidden)
            .flat_map(|row| {
                let activated = move |channel: usize| {
                    let term = |term: usize| {
                        columns(hidden)
                            .iter()
                            .enumerate()
                            .map(|(index, &column)| {
                                row[column].to_f32() * coefficient(term, channel, index).to_f32()
                            })
                            .sum::<f32>()
                    };
                    let gate = term(0);
                    let gate = bf16::from_f32(gate / (1.0 + (-gate).exp()));
                    let up = bf16::from_f32(term(1));
                    bf16::from_f32(gate.to_f32() * up.to_f32()).to_f32()
                };
                (0..hidden).map(move |output| {
                    (0..PARTS)
                        .map(|part| {
                            let term = activated(channel(intermediate, output, part))
                                * coefficient(2, output, part).to_f32();
                            (term, term.abs() / 128.0)
                        })
                        .fold((0.0, 0.0), |(sum, ulps), (term, ulp)| {
                            (sum + term, ulps + ulp)
                        })
                })
            })
            .collect()
    }

    fn weights<H: AxisName, I: AxisName>() -> (Vec<bf16>, Vec<bf16>)
    where
        Down<I>: AxisName,
    {
        let mut gate_up = vec![bf16::from_f32(0.0); GateUp::<I>::SIZE * H::SIZE];
        for channel in 0..I::SIZE {
            for term in 0..2 {
                for (index, column) in columns(H::SIZE).into_iter().enumerate() {
                    gate_up[(term * I::SIZE + channel) * H::SIZE + column] =
                        coefficient(term, channel, index);
                }
            }
        }
        let mut down = vec![bf16::from_f32(0.0); H::SIZE * Down::<I>::SIZE];
        for row in 0..H::SIZE {
            for part in 0..PARTS {
                down[row * Down::<I>::SIZE + channel(I::SIZE, row, part)] =
                    coefficient(2, row, part);
            }
        }
        (gate_up, down)
    }

    async fn mlp<H: AxisName, I: AxisName>(device: &mut Device) -> Mlp<H, I>
    where
        Down<I>: AxisName,
    {
        let (gate_up, down) = weights::<H, I>();
        Mlp::new(linear(device, gate_up).await, linear(device, down).await)
    }

    #[device(chip = 1)]
    fn leaves<H: AxisName, I: AxisName>(
        device: &mut Device,
        mlp: &crate::models::dense::Leaves<H, I>,
        rows: &Count,
        input: &HbmTensor<bf16, Chip, m![Sixteen, H]>,
    ) -> HbmTensor<bf16, Chip, m![Sixteen, H]>
    where
        crate::models::dense::Leaves<H, I>: for<'d, 'x> Module<
                (&'d mut Device, &'x Dm<Lanes<Sixteen>, Spread<H, Sixteen>>),
                Output = Dm<Lanes<Sixteen>, Spread<H, Sixteen>>,
            >,
    {
        let input: Dm<_, Spread<H, Sixteen>> = Dm {
            value: input.to_dm(&mut device.tdma),
            tiles: Live::<Sixteen>::read(rows),
        };
        mlp.forward((&mut *device, &input))
            .value
            .to_hbm(&mut device.tdma)
    }

    pub(super) async fn leaves_match_the_oracle<H: AxisName, I: AxisName>()
    where
        Down<I>: AxisName,
        crate::models::dense::Leaves<H, I>: for<'d, 'x> Module<
                (&'d mut Device, &'x Dm<Lanes<Sixteen>, Spread<H, Sixteen>>),
                Output = Dm<Lanes<Sixteen>, Spread<H, Sixteen>>,
            >,
    {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let (gate_up, down) = weights::<H, I>();
        let mlp: crate::models::dense::Leaves<H, I> = crate::Then(
            crate::ops::linear::Affine::new(
                HostTensor::<bf16, m![Term, I, H]>::from_vec(gate_up)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
                (),
            ),
            crate::Then(
                SwiGlu,
                crate::ops::linear::Affine::new(
                    HostTensor::<bf16, m![H, I]>::from_vec(down)
                        .to_hbm(&mut device.pdma)
                        .await
                        .unwrap(),
                    (),
                ),
            ),
        );
        let inputs = inputs(Sixteen::SIZE, H::SIZE);
        let expected = oracle(&inputs, H::SIZE, I::SIZE);
        let input: HbmTensor<bf16, Chip, m![Sixteen, H]> =
            HostTensor::<bf16, m![Sixteen, H]>::from_vec(inputs)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let lanes = Lanes::<Sixteen>::SIZE;
        let groups = <Groups<Sixteen> as M>::SIZE;
        for live in [
            Sixteen::SIZE,
            Sixteen::SIZE - lanes + 1,
            (groups - 1) * lanes,
            1,
        ] {
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let landed = launch(leaves, (&mut device, &mlp, &rows, &input))
                .await
                .unwrap()
                .to_host::<m![Sixteen, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, (got, &(expected, ulps))) in landed.iter().zip(&expected).enumerate() {
                let (row, column) = (index / H::SIZE, index % H::SIZE);
                let got = got.to_f32();
                assert!(
                    row >= live || (got - expected).abs() <= 0.03 * expected.abs().max(1.0) + ulps,
                    "{live} live rows, row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }

    async fn linear<In: AxisName, Out: AxisName>(
        device: &mut Device,
        weight: Vec<bf16>,
    ) -> Linear<In, Out> {
        Linear::new(Weight::new(
            HostTensor::<bf16, m![Out, In]>::from_vec(weight)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        ))
    }

    #[device(chip = 1)]
    fn projected<H: AxisName, I: AxisName, R: AxisName>(
        device: &mut Device,
        mlp: &Mlp<H, I>,
        rows: &Count,
        input: &HbmTensor<bf16, Chip, m![R, H]>,
        output: &mut HbmTensor<bf16, Chip, m![R, H]>,
    ) where
        Down<I>: AxisName,
        Mlp<H, I>: for<'d, 'x> Module<(&'d mut Device, (&'x Spread<H, R>, Live<R>)), Output = Spread<H, R>>,
    {
        let input: Spread<H, R> = input.to_dm(&mut device.tdma);
        mlp.forward((&mut *device, (&input, Live::read(rows))))
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// Every live row lands the oracle's row and every later row is untouched or its oracle row
    /// (a computed group may run past the live count), at every row live, at a partial last
    /// group, at all but one group and at none.
    async fn check<H: AxisName, I: AxisName, R: AxisName>()
    where
        Down<I>: AxisName,
        Mlp<H, I>: for<'d, 'x> Module<(&'d mut Device, (&'x Spread<H, R>, Live<R>)), Output = Spread<H, R>>,
    {
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let mlp = mlp::<H, I>(&mut device).await;
        let inputs = inputs(R::SIZE, H::SIZE);
        let expected = oracle(&inputs, H::SIZE, I::SIZE);
        let input = HostTensor::<bf16, m![R, H]>::from_vec(inputs)
            .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
            .await
            .unwrap();
        let lanes = Lanes::<R>::SIZE;
        let groups = <Groups<R> as M>::SIZE;
        for live in [R::SIZE, R::SIZE - lanes + 1, (groups - 1) * lanes, 1, 0] {
            let rows = count(live).to_hbm(&mut device.pdma).await.unwrap();
            let mut output = HostTensor::<bf16, m![R, H]>::zero()
                .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
                .await
                .unwrap();
            launch(projected, (&mut device, &mlp, &rows, &input, &mut output))
                .await
                .unwrap();
            let landed = output
                .to_host::<m![R, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, (got, &(expected, ulps))) in landed.iter().zip(&expected).enumerate() {
                let (row, column) = (index / H::SIZE, index % H::SIZE);
                let got = got.to_f32();
                let matches = (got - expected).abs() <= 0.03 * expected.abs().max(1.0) + ulps;
                assert!(
                    matches || (row >= live && got == 0.0),
                    "{live} live rows, row {row} column {column}: {got} vs {expected}",
                );
            }
        }
    }

    /// Each model width's MLP matches the oracle on sixteen rows.
    #[tokio::test(flavor = "current_thread")]
    async fn matches_the_oracle_at_every_width() {
        check::<Hidden<2560>, Intermediate<9728, 10240>, Sixteen>().await;
        check::<Hidden<4096>, Intermediate<12288>, Sixteen>().await;
        check::<Hidden<4096>, Intermediate<16384>, Sixteen>().await;
    }

    /// Above sixteen rows every intermediate tile reaches the sum, the fourth of a 16384-wide MLP
    /// included, and the row groups stay apart.
    #[tokio::test(flavor = "current_thread")]
    async fn sums_every_intermediate_tile_above_sixteen_rows() {
        check::<Hidden<4096>, Intermediate<16384>, Rows<32>>().await;
        check::<Hidden<4096>, Intermediate<16384>, Rows<64>>().await;
        check::<Hidden<4096>, Intermediate<12288>, Rows<32>>().await;
        check::<Hidden<4096>, Intermediate<12288>, Rows<64>>().await;
    }

    type RowsMlp<H, I> =
        crate::modules::feedforward::Mlp<Linear<H, GateUp<I>>, SwiGlu, Linear<I, H>>;

    #[device(chip = 1)]
    fn gated_rows<I: AxisName, R: AxisName>(
        device: &mut Device,
        input: &HbmTensor<bf16, Chip, m![R, { Symbol<GateUp<I>> }]>,
    ) -> HbmTensor<bf16, Chip, m![R, I]> {
        SwiGlu.forward((device, input))
    }

    #[device(chip = 1)]
    fn mlp_rows<H: AxisName, I: AxisName, R: AxisName>(
        device: &mut Device,
        mlp: &RowsMlp<H, I>,
        input: &HbmTensor<bf16, Chip, m![R, H]>,
    ) -> HbmTensor<bf16, Chip, m![R, H]> {
        mlp.forward((device, input))
    }

    impl Fixture {
        fn silu(x: f64) -> f64 {
            x / (1.0 + (-x).exp())
        }

        fn rounded(x: f64) -> f64 {
            f64::from(bf16::from_f32(x as f32).to_f32())
        }
    }

    /// Rows in HBM take `bf16(bf16(silu(gate)) * up)` of their two halves, the SiLU rounded first.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_in_hbm_round_the_silu_first() {
        Fixture::gated_rows_match::<Rows<128>>(1).await;
    }

    /// `R` gate/up rows in HBM, every `stride`-th row checked against the reference with the SiLU
    /// rounded to BF16 first.
    impl Fixture {
        async fn gated_rows_match<R: AxisName>(stride: usize) {
            let _lease = crate::tests::DEVICE.lock().await;
            type I = Intermediate<14336>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let width = I::SIZE;
            let packed: Vec<bf16> = crate::ops::normalization::tests::random(
                R::SIZE * 2 * width,
                0x2545_f491_4f6c_dd1d,
            )
            .iter()
            .map(|v| bf16::from_f32(v.to_f32() * 6.0))
            .collect();
            let input = HostTensor::<bf16, m![R, { Symbol<GateUp<I>> }]>::from_vec(packed.clone())
                .to_hbm::<Chip, m![R, { Symbol<GateUp<I>> }]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(gated_rows, (&mut device, &input))
                .await
                .unwrap()
                .to_host::<m![R, I]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            for (index, got) in output.iter().enumerate() {
                let (row, column) = (index / width, index % width);
                if row % stride != 0 {
                    continue;
                }
                let gate = f64::from(packed[row * 2 * width + column].to_f32());
                let up = f64::from(packed[row * 2 * width + width + column].to_f32());
                let expected = Self::rounded(Self::silu(gate)) * up;
                let got = f64::from(got.to_f32());
                assert!(
                    (got - expected).abs() <= expected.abs() / 128.0 + 1e-3,
                    "row {row} column {column}: {got} vs {expected}"
                );
            }
        }
    }
    /// The MLP over `R` rows in HBM: `checked` rows match the reference with BF16 gate, up, SiLU
    /// and product, within 2 % of each row's RMS.
    impl Fixture {
        async fn rows_mlp_matches<R: AxisName>(checked: &[usize]) {
            let _lease = crate::tests::DEVICE.lock().await;
            type H = Hidden<5376>;
            type I = Intermediate<14336>;
            let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
            let random = crate::ops::normalization::tests::random;
            let (hidden, width) = (H::SIZE, I::SIZE);
            let inputs = random(R::SIZE * hidden, 0x2545_f491_4f6c_dd1d);
            let up_weight = random(2 * width * hidden, 0x9e37_79b9_7f4a_7c15);
            let down_weight = random(hidden * width, 0x0123_4567_89ab_cdef);
            let widened = |values: &[bf16]| values.iter().map(|v| v.to_f32()).collect::<Vec<f32>>();
            let (x, w1, w2) = (widened(&inputs), widened(&up_weight), widened(&down_weight));
            let up: Linear<H, GateUp<I>> = Linear::new(Weight::new(
                HostTensor::<bf16, m![{ Symbol<GateUp<I>> }, H]>::from_vec(up_weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let down: Linear<I, H> = Linear::new(Weight::new(
                HostTensor::<bf16, m![H, I]>::from_vec(down_weight)
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ));
            let mlp: RowsMlp<H, I> = crate::Then(up, crate::Then(SwiGlu, down));
            let input = HostTensor::<bf16, m![R, H]>::from_vec(inputs)
                .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
                .await
                .unwrap();
            let output = launch(mlp_rows, (&mut device, &mlp, &input))
                .await
                .unwrap()
                .to_host::<m![R, H]>(&mut device.pdma)
                .await
                .unwrap()
                .into_vec();
            let dot = |a: &[f32], b: &[f32]| -> f64 {
                a.iter()
                    .zip(b)
                    .map(|(a, b)| f64::from(*a) * f64::from(*b))
                    .sum()
            };
            for &row in checked {
                let x = &x[row * hidden..][..hidden];
                let product: Vec<f32> = (0..width)
                    .map(|j| {
                        let gate = Self::rounded(dot(x, &w1[j * hidden..][..hidden]));
                        let up = Self::rounded(dot(x, &w1[(width + j) * hidden..][..hidden]));
                        Self::rounded(Self::rounded(Self::silu(gate)) * up) as f32
                    })
                    .collect();
                let expected: Vec<f64> = (0..hidden)
                    .map(|o| dot(&product, &w2[o * width..][..width]))
                    .collect();
                let rms = (expected.iter().map(|e| e * e).sum::<f64>() / hidden as f64).sqrt();
                for (column, expected) in expected.iter().enumerate() {
                    let got = f64::from(output[row * hidden + column].to_f32());
                    assert!(
                        (got - expected).abs() <= 0.02 * rms,
                        "row {row} column {column}: {got} vs {expected} (row rms {rms})"
                    );
                }
            }
        }
    }
    /// Four rows at the ends of the first two tiles through the whole MLP.
    #[tokio::test(flavor = "current_thread")]
    async fn rows_mlp_matches_the_reference() {
        Fixture::rows_mlp_matches::<Rows<128>>(&[0, 63, 64, 127]).await;
    }

    /// The 25600-wide MLP, whose gate/up pair passes through HBM, matches the oracle.
    #[tokio::test(flavor = "current_thread")]
    async fn wide_mlp_matches_the_oracle() {
        check::<Hidden<5120>, Intermediate<25600>, Sixteen>().await;
    }
}
