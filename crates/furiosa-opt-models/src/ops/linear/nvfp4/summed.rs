//! One BF16 row projected by NVFP4 weights under the experimental summed contract, without
//! restoring them: the fetch table lookup decodes each E2M1 code straight to BF16 into the
//! contraction, which sums every 16-input block; the vector engine multiplies each block sum by its
//! E4M3 scale, adds a row's blocks, divides by the tensor scale and adds the slices that hold parts
//! of the row.

use super::Vector;
use crate::ops::linear::block_einsum::scaled_block_sums;
use crate::tensor::form::BlockScaled;
use crate::tensor::form::nvfp4::{Held, Scale, Scales, Summed, W4A16};
use crate::tensor::memory::Memory;
use crate::{
    Op,
    tensor::{parallel::Chip, scalar},
};
use core::borrow::Borrow;
use furiosa_opt_std::prelude::*;

/// Ways a row's inputs split over slices: the fewest, a power of two, that keep a slice's part
/// within 4096 inputs, so its four rows' FP32 block scales take at most 4 KiB of the VRF.
#[derive(Clone, Debug)]
struct Parts<I>(std::marker::PhantomData<I>);

impl<I: AxisName> AxisName for Parts<I> {
    const NAME: Ident = Ident::new("Parts");
    const SIZE: usize = {
        let mut parts = 1;
        while I::SIZE / parts > 4096 {
            parts *= 2;
        }
        parts
    };
}

/// Blocks each cluster keeps: as many as fill 128 of its slices with the parts of their rows.
#[derive(Clone, Debug)]
struct Blocks<I>(std::marker::PhantomData<I>);

impl<I: AxisName> AxisName for Blocks<I> {
    const NAME: Ident = Ident::new("Blocks");
    const SIZE: usize = 128 / <Parts<I> as AxisName>::SIZE;
}

/// Rows of a block: each cluster keeps half of the outputs in its blocks.
#[derive(Clone, Debug)]
struct Rows<I, O>(std::marker::PhantomData<(I, O)>);

impl<I: AxisName, O: AxisName> AxisName for Rows<I, O> {
    const NAME: Ident = Ident::new("Rows");
    const SIZE: usize = O::SIZE / 2 / <Blocks<I> as AxisName>::SIZE;
}

type Block<I, O> = Symbol<Rows<I, O>>;
type Ways<I> = Symbol<Parts<I>>;
/// The first half of the outputs in cluster 0, the second in cluster 1.
type Cluster<O> = m![O / (O / 2)];
/// A slice keeps one part of the rows of one block; the parts of a block are adjacent, innermost,
/// for the inter-slice reducer.
type Slices<I, O> = m![O % (O / 2) / ({ Block<I, O> }), 1 # 2, I / (I / ({ Ways<I> }))];
type Reduced<I, O> = m![O % (O / 2) / ({ Block<I, O> }), 1 # 2, 1 # ({ Ways<I> })];
/// A slice's part of a row's inputs.
type Part<I> = m![I % (I / ({ Ways<I> }))];
/// The four rows of every block one pass covers: the transpose to the output packet takes at
/// most four, and a commit packet at least eight bytes.
type Elements<I, O> = m![O % ({ Block<I, O> }) = 4];
/// A pass's block sums: its rows, the 16-input blocks of a slice's part, and the `1` of the input's
/// TRF element, which the contraction keeps.
type Sums<I, O> = m![{ Elements<I, O> }, { Part<I> } / 16, 1];

/// One contiguous BF16 row through a weight under the summed contract.
impl<I: AxisName, O: AxisName, K, V, Y: Memory> Op<&Vector<I>>
    for BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16, Y>, Chip, K, V>
where
    K: Borrow<HbmTensor<f4e2m1, Chip, m![O, I]>>,
    V: Borrow<HbmTensor<f8e4m3, Chip, Scales<I, O>>>,
    Held<Y>: Borrow<Scale>,
{
    type Output = Vector<O>;

    /// The codes decode to BF16 straight into the contraction, and the scales apply to the block
    /// sums, so no tile is restored.
    fn forward(&self, device: &mut Device, input: &Vector<I>) -> Self::Output {
        let global: &Scale = self.contract.global.borrow();
        project(device, self.codes(), self.scales(), global, input.view())
    }
}

/// Every output `bf16(sum_b scale_b (sum_{k in b} code_k x_k) / global)`, the sums in FP32 and the
/// result rounded once. Each pass lands four rows of every block; the block scales land once.
fn project<I: AxisName, O: AxisName>(
    device: &mut Device,
    codes: &HbmTensor<f4e2m1, Chip, m![O, I]>,
    scales: &HbmTensor<f8e4m3, Chip, Scales<I, O>>,
    global: &Scale,
    input: HbmTensorView<'_, bf16, Chip, m![I]>,
) -> HbmTensor<bf16, Chip, m![O]> {
    const {
        assert!(I::SIZE.is_multiple_of(<Parts<I> as AxisName>::SIZE * 128));
        assert!(O::SIZE.is_multiple_of(2 * <Blocks<I> as AxisName>::SIZE * 4));
    };
    // Every slice of a block's part holds that part of the row, in both clusters.
    let input: DmTensor<
        bf16,
        Chip,
        m![2],
        m![{ Symbol<Blocks<I>> }, 1 # 2, I / (I / ({ Ways<I> }))],
        Part<I>,
    > = input.to_dm(&mut device.tdma);
    // SAFETY: the copies of a block's part of the row sit where its slices are.
    let input: DmTensor<bf16, Chip, Cluster<O>, Slices<I, O>, Part<I>> = unsafe { input.reshape() };
    let input: TrfTensor<bf16, Chip, Cluster<O>, Slices<I, O>, m![1], m![1, { Part<I> }]> = device
        .sub
        .begin(input.view())
        .fetch::<m![{ Part<I> } / 16], m![I % 16]>()
        .collect::<m![{ Part<I> } / 16], m![I % 16]>()
        .to_trf();
    let global: DmTensor<f32, Chip, Cluster<O>, Slices<I, O>, m![1 # 8]> =
        global.to_dm(&mut device.tdma);
    let global = scalar::register(device, &global);
    let scales: DmTensor<
        f8e4m3,
        Chip,
        Cluster<O>,
        Slices<I, O>,
        m![O % ({ Block<I, O> }), { Part<I> } / 16],
    > = scales.to_dm(&mut device.tdma);
    let mut output: DmTensor<bf16, Chip, Cluster<O>, Reduced<I, O>, m![O % ({ Block<I, O> })]> =
        DmTensor::new();
    #[unroll]
    for pass in 0..(<Rows<I, O> as AxisName>::SIZE / 4) {
        let at = 4 * pass;
        let scale: VrfTensor<f32, Chip, Cluster<O>, Slices<I, O>, m![{ Elements<I, O> }, { Part<I> } / 16]> = device
            .sub
            .begin(scales.view().tile::<m![O % ({ Block<I, O> })], 4, m![{ Elements<I, O> } # ({ Block<I, O> }), { Part<I> } / 16]>(at))
            .fetch::<Elements<I, O>, m![{ Part<I> } / 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Elements<I, O> }, { Part<I> } / 128], m![I / 16 % 8]>()
            .to_vrf();
        let codes: DmTensor<f4e2m1, Chip, Cluster<O>, Slices<I, O>, m![{ Elements<I, O> }, { Part<I> }]> = codes
            .view()
            .tile::<m![O % ({ Block<I, O> })], 4, m![O / (O / 2), O % (O / 2) / ({ Block<I, O> }), { Elements<I, O> } # ({ Block<I, O> }), I]>(at)
            .to_dm(&mut device.tdma);
        scaled_block_sums::<I, Elements<I, O>, Part<I>, Sums<I, O>, _, _, _, _>(device, codes.view(), &input, &scale)
            .vector_intra_slice_reduce::<I, m![{ Elements<I, O> }, 1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_fp_div(&global)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_inter_slice_reduce::<Reduced<I, O>, m![{ Elements<I, O> }, 1]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![1 # 16]>()
            .transpose::<m![1], m![{ Elements<I, O> } # 16]>()
            .commit_trim::<Elements<I, O>>()
            .commit_view(
                output
                    .view_mut()
                    .tile::<m![O % ({ Block<I, O> })], 4, m![{ Elements<I, O> } #{!} ({ Block<I, O> })]>(at),
            );
    }
    output.to_hbm(&mut device.tdma)
}

#[cfg(test)]
mod tests {
    use crate::Op;
    use crate::tensor::form::BlockScaled;
    use crate::tensor::form::nvfp4::{Scales, Summed, W4A16};
    use crate::tensor::parallel::Chip;
    use core::marker::PhantomData;
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;
    use std::iter;

    axes![Hidden = 3840, Intermediate = 15360];

    type Coded<I, O> = BlockScaled<I, O, f4e2m1, f8e4m3, Scales<I, O>, Summed<W4A16>>;

    #[device(chip = 1)]
    fn projected<I: AxisName, O: AxisName>(
        device: &mut Device,
        linear: &Coded<I, O>,
        input: &HbmTensor<bf16, Chip, m![I]>,
    ) -> HbmTensor<bf16, Chip, m![O]> {
        linear.forward(device, input)
    }

    /// Uniform values in `[0, 1)` from a xorshift state.
    fn uniform(count: usize, mut state: u64) -> impl Iterator<Item = f64> {
        (0..count).map(move |_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
    }

    /// Random codes, E4M3 scales from 8 to 448 and a row in `[-2, 2)` project within one rounding
    /// to BF16 of the FP64 product, plus the FP32 sums' error: `2^-9` of the outputs' RMS.
    async fn rounds_the_product<I: AxisName, O: AxisName>(device: &mut Device, seed: u64) {
        let (i, o) = (I::SIZE, O::SIZE);
        let codes: Vec<f4e2m1> = uniform(o * i, seed)
            .map(|u| f4e2m1::from_bits((u * 16.0) as u8))
            .collect();
        let scales: Vec<f8e4m3> = uniform(o * i / 16, seed + 1)
            .map(|u| f8e4m3::from_f32((8.0 + 440.0 * u) as f32))
            .collect();
        let global = 2f32.powi(14) * 1.37;
        let row: Vec<bf16> = uniform(i, seed + 2)
            .map(|u| bf16::from_f32((4.0 * u - 2.0) as f32))
            .collect();
        let exact: Vec<f64> = (0..o)
            .map(|r| {
                (0..i)
                    .map(|c| {
                        let weight = f64::from(codes[r * i + c].to_f32())
                            * f64::from(scales[(r * i + c) / 16].to_f32())
                            / f64::from(global);
                        weight * f64::from(f32::from(row[c]))
                    })
                    .sum()
            })
            .collect();
        let linear: Coded<I, O> = BlockScaled {
            weight: HostTensor::<f4e2m1, m![O, I]>::from_vec(codes)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            weight_scale: HostTensor::<f8e4m3, Scales<I, O>>::from_vec(scales)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            contract: Summed {
                input: W4A16,
                global: HostTensor::<f32, m![1 # 8]>::from_vec(iter::once(global).chain([0.0; 7]))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            },
            shape: PhantomData,
        };
        let input = HostTensor::<bf16, m![I]>::from_vec(row)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let output = launch(projected, (&mut *device, &linear, &input))
            .await
            .unwrap()
            .to_host::<m![O]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let rms = (exact.iter().map(|e| e * e).sum::<f64>() / o as f64).sqrt();
        for (at, (got, want)) in output.iter().zip(&exact).enumerate() {
            let got = f64::from(f32::from(*got));
            assert!(
                (got - want).abs() <= want.abs() / 256.0 + rms / 512.0,
                "{}x{} output {at}: {got} for {want}",
                I::SIZE,
                O::SIZE
            );
        }
    }

    /// Gemma-4 12B's MLP projections: the up and gate projections keep a row's inputs whole in a
    /// slice, the down projection splits them in four.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_gemma4_mlp_rows() {
        let mut device = Device::new(projected.topology()).unwrap();
        rounds_the_product::<Hidden, Intermediate>(&mut device, 1).await;
        rounds_the_product::<Intermediate, Hidden>(&mut device, 4).await;
    }
}
