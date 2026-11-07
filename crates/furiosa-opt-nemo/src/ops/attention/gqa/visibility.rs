use super::PagedAttention;
use crate::Module;
use crate::compose::Space;
use crate::ops::layout::Dm;
use furiosa_opt_std::prelude::*;

pub trait Mapping: Space {
    type Rows: M;
    type Groups: M;
    type Keys: M;
}

/// Query/key coordinates whose positive excess marks an invisible key.
pub trait Causal: Space {
    type Time: M;
    type Packet: M;
    type Tags: M;
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, E: M, S: Causal>
    Module<(&mut Device, Dm<S, &'x DmTensor<bf16, B, C, L, E>>)>
    for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, VrfTensor<i4, B, C, L, S::Tags>>;

    fn forward(
        &self,
        (device, input): (&mut Device, Dm<S, &'x DmTensor<bf16, B, C, L, E>>),
    ) -> Self::Output {
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(input.value.view())
                .fetch::<S::Time, S::Packet>()
                .fetch_cast::<f32>()
                .collect::<S::Time, S::Packet>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Comparison([
                    Cmp::Greater(0.0f32),
                    Cmp::False,
                    Cmp::False,
                    Cmp::False,
                ]))
                .vector_log(&mut device.sub),
        }
    }
}

impl<'x, H, G, D, const SCALE: usize, B: M, C: M, L: M, P: M, N: M, S: Mapping>
    Module<(
        &mut Device,
        Dm<
            S,
            (
                DmTensorView<'x, f32, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>,
                DmTensorView<'x, f32, B, C, L, P>,
                DmTensorView<'x, f32, B, C, L, N>,
            ),
        >,
    )> for PagedAttention<H, G, D, SCALE, B>
{
    type Output = Dm<S, DmTensor<bf16, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]>>;

    fn forward(
        &self,
        (device, input): (
            &mut Device,
            Dm<
                S,
                (
                    DmTensorView<
                        'x,
                        f32,
                        B,
                        C,
                        L,
                        m![{ S::Rows }, { S::Groups }, { S::Keys } / 16],
                    >,
                    DmTensorView<'x, f32, B, C, L, P>,
                    DmTensorView<'x, f32, B, C, L, N>,
                ),
            >,
        ),
    ) -> Self::Output {
        let (scores, positions, counts) = input.value;
        let count: VrfTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, 1 # 8]> = device
            .sub
            .begin(counts)
            .fetch::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }], m![1 # 8]>()
            .to_vrf();
        let beyond: DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]> = device
            .main
            .begin(positions)
            .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 64], m![{ S::Keys } / 16 % 4]>()
            .vector_fp_binary(FpBinaryOp::SubF, &count)
            .vector_widen_concat::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .vector_final()
            .commit_trim::<m![{ S::Keys } / 16 % 8]>()
            .commit();
        let tags: VrfTensor<i4, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]> = device
            .main
            .begin(beyond.view())
            .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>(
            )
            .vector_init()
            .vector_intra_slice_tag(TagMode::Comparison([
                Cmp::Greater(-0.5f32),
                Cmp::False,
                Cmp::False,
                Cmp::False,
            ]))
            .vector_log(&mut device.sub);
        let scaled: DmTensor<f32, B, C, L, m![{ S::Rows }, { S::Groups }, { S::Keys } / 16]> = device
            .main
            .begin(scores)
            .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
            .vector_init()
            .vector_tagged(&tags)
            .vector_scale(Branched::imm(
                TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]),
                f32::from_bits(SCALE as u32),
            ))
            .vector_final()
            .commit_trim::<m![{ S::Keys } / 16 % 8]>()
            .commit();
        Dm {
            tiles: input.tiles,
            value: device
                .main
                .begin(scaled.view())
                .fetch::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .collect::<m![{ S::Rows }, { S::Groups }, { S::Keys } / 128], m![{ S::Keys } / 16 % 8]>()
                .vector_init()
                .vector_tagged(&tags)
                .vector_logic_with_mode(
                    LogicBinaryOpF32::BitAnd,
                    BinaryArgMode::Mode11,
                    Branched::imm(
                        TagGuard::matches([
                            BitReq::One,
                            BitReq::Ignore,
                            BitReq::Ignore,
                            BitReq::Ignore,
                        ]),
                        f32::from_bits(0xff7f_0000),
                    ),
                )
                .vector_final()
                .cast::<bf16, m![{ S::Keys } / 16 % 8 # 16]>()
                .commit_trim::<m![{ S::Keys } / 16 % 8]>()
                .commit(),
        }
    }
}
