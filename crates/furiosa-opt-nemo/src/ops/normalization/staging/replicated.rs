use super::*;

pub struct Inverse<V>(V);

impl<W: AxisName, const E: usize, C: M, Cl: M, S: M, L: M>
    RmsNorm<W, E, C, DmTensorView<'_, bf16, C, Cl, S, L>>
{
    pub fn gain<Time: M, Packet: M>(&self, device: &mut Device) -> VrfTensor<f32, C, Cl, S, L> {
        device
            .sub
            .begin(self.weight().clone())
            .fetch::<Time, Packet>()
            .fetch_cast::<f32>()
            .collect::<Time, Packet>()
            .to_vrf()
    }

    pub fn replicated<
        'a,
        T: Space,
        R: M,
        Features: M,
        Copies: M,
        Trim: M,
        const STASH: bool,
        Input: M,
    >(
        &'a self,
        device: &mut Device,
        input: &'a Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
    ) -> Statistics<
        'a,
        Self,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Total<Sums<C, Cl, S, R>>,
    > {
        const {
            assert!(W::SIZE.is_multiple_of(L::SIZE));
            assert!(L::SIZE == Features::SIZE);
            assert!(Features::SIZE.is_multiple_of(8));
        }
        let value = device
            .main
            .begin(input.value.clone())
            .fetch::<m![{ R }, { Features } / 8], m![{ Features } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ R }, { Features } / 8], m![{ Features } % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![{ R }, { Features } / 4], m![{ Features } % 4]>();
        macro_rules! reduce {
            ($value:expr) => {
                $value
                    .vector_intra_slice_reduce::<W, R, m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(const { W::SIZE as f32 })
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_clip(ClipBinaryOpF32::Add, const { f32::from_bits(E as u32) / (W::SIZE / L::SIZE) as f32 })
                    .vector_inter_slice_reduce::<Copies, R>(InterSliceReduceOpF32::Add)
                    .vector_final()
                    .commit_trim::<Trim>()
                    .commit()
            };
        }
        let sums: Sums<C, Cl, Copies, R> = if STASH {
            reduce!(
                value
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
            )
        } else {
            reduce!(value.vector_fp_binary_with_mode(
                FpBinaryOp::MulF(FpMulAlu::Mul0),
                BinaryArgMode::Mode00,
                0.0_f32,
            ))
        };
        Statistics {
            norm: self,
            input,
            // SAFETY: the reduction replicates each row scalar in the original feature-slice order.
            value: Total(unsafe { sums.reshape() }),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M> Statistics<'a, N, I, Total<Sums<C, Cl, S, R>>> {
    pub fn inverse(
        self,
        device: &mut Device,
    ) -> Statistics<'a, N, I, Inverse<VrfTensor<f32, C, Cl, S, m![{ R }, 1 # 8]>>> {
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Inverse(
                device
                    .main
                    .begin(self.value.0.view())
                    .fetch::<R, m![1 # 2]>()
                    .collect::<R, m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_trim::<m![1 # 4]>()
                    .vector_fp_unary(FpUnaryOp::Sqrt)
                    .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_final()
                    .to_vrf(&mut device.sub),
            ),
        }
    }
}

impl<'a, 'w, W: AxisName, const E: usize, C: M, Cl: M, S: M, L: M, T: Space, R: M, Input: M>
    Statistics<
        'a,
        RmsNorm<W, E, C, DmTensorView<'w, bf16, C, Cl, S, L>>,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Inverse<VrfTensor<f32, C, Cl, S, m![{ R }, 1 # 8]>>,
    >
{
    pub fn apply<'l, Time: M, Packet: M, Narrow: M>(
        self,
        device: &'l mut Device,
        gain: &VrfTensor<f32, C, Cl, S, L>,
    ) -> Dm<
        T,
        VectorFpTensor<
            'l,
            { Tu::Main },
            f32,
            C,
            Cl,
            S,
            Narrow,
            m![{ Packet } % 4],
            Fresh,
            { stage::VeOrder::IntraFirst },
        >,
    >
    where
        'a: 'l,
    {
        Dm {
            tiles: self.input.tiles,
            value: device
                .main
                .begin(self.input.value.clone())
                .fetch::<Time, Packet>()
                .fetch_cast::<f32>()
                .collect::<Time, Packet>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<Narrow, m![{ Packet } % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self.value.0)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), gain),
        }
    }
}
