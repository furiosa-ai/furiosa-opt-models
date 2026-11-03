//! Exact NVFP4 restoration shared by tiled projections. Each local feature run consists of
//! whole 16-feature blocks; BF16 holds E2M1 code times E4M3 block scale exactly.
use super::Scales;
use crate::ops::vector::Restore as _;
use crate::tensor::form::nvfp4::Scale;
use crate::tensor::parallel::Chip;
use furiosa_opt_std::prelude::*;

/// Land a scale once per cluster, then broadcast on the switch ring. A direct
/// replicated DMA issues hundreds of tiny transfers for each scalar.
pub(crate) fn scalar<C: M, S: M>(
    device: &mut Device,
    value: &Scale,
) -> VrfTensor<f32, Chip, C, S, m![1 # 8]> {
    const {
        assert!(C::SIZE == 2 && matches!(S::SIZE, 32 | 256));
    }
    if const { S::SIZE == 32 } {
        let landed: DmTensor<f32, Chip, C, S, m![1 # 8]> = value.to_dm(&mut device.tdma);
        return device
            .sub
            .begin(landed.view())
            .fetch::<m![1], m![1 # 8]>()
            .collect::<m![1], m![1 # 8]>()
            .to_vrf();
    }
    let landed: DmTensor<f32, Chip, m![2], m![1 # 256], m![1 # 8]> = value.to_dm(&mut device.tdma);
    let broadcast: VrfTensor<f32, Chip, m![2], m![256], m![1 # 8]> = device
        .sub
        .begin(landed.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<m![256], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    // SAFETY: every physical slice of both clusters carries the same packet;
    // the checked geometry only renames these cluster/slice owners.
    unsafe { broadcast.reshape() }
}

/// Apply the tensor-scale product once, after every contraction partial has accumulated in FP32.
pub(crate) fn finish<C: M, S: M, E: M + 'static>(
    device: &mut Device,
    scales: &Scales,
    input: DmTensor<f32, Chip, C, S, E>,
) -> DmTensor<bf16, Chip, C, S, E> {
    let factor = scalar::<C, S>(device, &scales.output);
    // SAFETY: flatten the same contiguous local elements without changing chip/slice ownership.
    let input = unsafe { input.reshape::<Chip, C, S, m![{ Symbol<Flat<E>> }]>() };
    let output: DmTensor<bf16, Chip, C, S, m![{ Symbol<Flat<E>> }]> = device
        .main
        .begin(input.view())
        .fetch::<m![{ Symbol<Flat<E>> } / 8], m![{ Symbol<Flat<E>> } % 8]>()
        .collect::<m![{ Symbol<Flat<E>> } / 8], m![{ Symbol<Flat<E>> } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Symbol<Flat<E>> } / 4], m![{ Symbol<Flat<E>> } % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &factor)
        .vector_widen_concat::<m![{ Symbol<Flat<E>> } / 8], m![{ Symbol<Flat<E>> } % 8]>()
        .vector_final()
        .cast::<bf16, m![{ Symbol<Flat<E>> } % 8 # 16]>()
        .commit_trim::<m![{ Symbol<Flat<E>> } % 8]>()
        .commit();
    // SAFETY: restore the original local axes after the elementwise scale and cast.
    unsafe { output.reshape() }
}

use core::marker::PhantomData;
#[derive(Clone, Debug)]
pub(crate) struct Flat<E>(PhantomData<E>);
impl<E: M + 'static> AxisName for Flat<E> {
    const NAME: Ident = Ident::new("Flat");
    const SIZE: usize = E::SIZE;
}
type Blocks<E> = m![{ Symbol<Flat<E>> } / 16];
type Register<E, C, S> = VrfTensor<f32, Chip, C, S, Blocks<E>>;

pub(crate) fn restore<C: M, S: M, E: M + 'static, B: M>(
    device: &mut Device,
    codes: DmTensor<f4e2m1, Chip, C, S, E>,
    scales: DmTensor<f8e4m3, Chip, C, S, B>,
) -> DmTensor<bf16, Chip, C, S, E> {
    restore_view(device, codes.view(), scales.view())
}

pub(crate) fn restore_view<C: M, S: M, E: M + 'static, B: M>(
    device: &mut Device,
    codes: DmTensorView<'_, f4e2m1, Chip, C, S, E>,
    scales: DmTensorView<'_, f8e4m3, Chip, C, S, B>,
) -> DmTensor<bf16, Chip, C, S, E> {
    const {
        assert!(E::SIZE == B::SIZE * 16 && B::SIZE * 4 <= 8192);
    }
    // SAFETY: flattening preserves each local row's consecutive 16-feature blocks.
    let codes = unsafe { codes.reshape::<Chip, C, S, m![{ Symbol<Flat<E>> }]>() };
    let scales = unsafe { scales.reshape::<Chip, C, S, Blocks<E>>() };
    let scales: Register<E, C, S> = device
        .sub
        .begin(scales)
        .fetch::<m![{ Blocks<E> } / 8], m![{ Blocks<E> } % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![{ Blocks<E> } / 8], m![{ Blocks<E> } % 8]>()
        .to_vrf();
    let restored: DmTensor<bf16, Chip, C, S, m![{ Symbol<Flat<E>> }]> = device
        .main
        .begin(codes)
        .fetch::<m![{ Symbol<Flat<E>> } / 32], m![{ Symbol<Flat<E>> } % 32]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![{ Symbol<Flat<E>> } / 8], m![{ Symbol<Flat<E>> } % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![{ Symbol<Flat<E>> } / 4], m![{ Symbol<Flat<E>> } % 4]>()
        .vector_restore::<m![{ Symbol<Flat<E>> } / 8], m![{ Symbol<Flat<E>> } % 8], _, _>(
            &scales,
            |stream| stream,
        )
        .commit();
    // SAFETY: inverse of the local flattening above.
    unsafe { restored.reshape() }
}
