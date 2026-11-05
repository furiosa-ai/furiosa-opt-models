use super::*;

impl<V, Level> Exchange<V, Level> {
    pub fn values(&self) -> (&V, &V) {
        (&self.0, &self.1)
    }
    pub fn resume<'a, N, I>(self, norm: &'a N, input: &'a I) -> Statistics<'a, N, I, Self> {
        Statistics {
            norm,
            input,
            value: self,
        }
    }
}

impl<'a, N, I, V> Statistics<'a, N, I, V> {
    pub fn into_value(self) -> V {
        self.value
    }
}

impl<V> Gathered<V> {
    pub fn resume<'a, N, I>(self, norm: &'a N, input: &'a I) -> Statistics<'a, N, I, Self> {
        Statistics {
            norm,
            input,
            value: self,
        }
    }
}

impl<V> Total<V> {
    pub fn resume<'a, N, I>(self, norm: &'a N, input: &'a I) -> Statistics<'a, N, I, Self> {
        Statistics {
            norm,
            input,
            value: self,
        }
    }
}

impl<W: AxisName, const E: usize, C: M, Cl: M, S: M, L: M>
    RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>>
{
    pub fn total<'a, T: Space, R: M, Input: M>(
        &'a self,
        device: &mut Device,
        (input, (local, peer)): (
            &'a Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
            (&Sums<C, Cl, S, R>, &Sums<C, Cl, S, R>),
        ),
    ) -> Statistics<
        'a,
        Self,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Total<Sums<C, Cl, S, R>>,
    > {
        axes![Term = 2];
        let total = device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(local.view(), peer.view())
            .fetch::<m![{ R }, Term], m![1 # 8]>()
            .collect::<m![{ R }, Term], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, R>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        self.resume(input, total)
    }
    /// Carried sums must belong to this norm and the same committed input snapshot.
    pub fn resume<'a, T: Space, R: M, Input: M>(
        &'a self,
        input: &'a Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        sums: Sums<C, Cl, S, R>,
    ) -> Statistics<
        'a,
        Self,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Total<Sums<C, Cl, S, R>>,
    > {
        Statistics {
            norm: self,
            input,
            value: Total(sums),
        }
    }
}

impl<C: M, Cl: M, S: M, R: M, Level> Exchange<Sums<C, Cl, S, R>, Level> {
    pub fn combine_into(&self, device: &mut Device, output: &mut Sums<C, Cl, S, R>) {
        axes![Term = 2];
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(self.0.view(), self.1.view())
            .fetch::<m![{ R }, Term], m![1 # 8]>()
            .collect::<m![{ R }, Term], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, R>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit_view(output.view_mut());
    }

    pub fn chips_into(
        &self,
        device: &mut Device,
        (local, peer): (&mut Sums<C, Cl, S, R>, &mut Sums<C, Cl, S, R>),
    ) {
        const {
            assert!(C::SIZE == 2);
        }
        self.combine_into(device, local);
        local
            .chip_shuffle([1, 0])
            .to_dm_view(&mut device.tdma, peer.view_mut());
    }
}

impl<'a, W: AxisName, const E: usize, C: M, Cl: M, S: M, L: M, T: Space, R: M, Input: M>
    Statistics<
        'a,
        RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>>,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Total<Sums<C, Cl, S, R>>,
    >
{
    pub fn complete<'l, Groups: M, Features: M>(
        self,
        device: &'l mut Device,
    ) -> Dm<
        T,
        VectorFpTensor<
            'l,
            { Tu::Main },
            f32,
            C,
            Cl,
            S,
            m![{ Groups }, { R }, { Features } / 4],
            m![{ Features } % 4],
            Fresh,
            { stage::VeOrder::IntraFirst },
        >,
    >
    where
        'a: 'l,
    {
        let inverse: VrfTensor<f32, C, Cl, S, m![{ R }, 1 # 8]> = device
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
            .to_vrf::<m![{ R }, 1 # 8]>(&mut device.sub);
        let gain: VrfTensor<f32, C, Cl, S, L> = device
            .sub
            .begin(self.norm.weight().view())
            .fetch::<m![{ Groups }, { Features } / 8], m![{ Features } % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Groups }, { Features } / 8], m![{ Features } % 8]>()
            .to_vrf();
        Dm {
            tiles: self.input.tiles,
            value: device
                .main
                .begin(self.input.value.clone())
                .fetch::<m![{ Groups }, { R }, { Features } / 8], m![{ Features } % 8]>()
                .fetch_cast::<f32>()
                .collect::<m![{ Groups }, { R }, { Features } / 8], m![{ Features } % 8]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_narrow_split::<m![{ Groups }, { R }, { Features } / 4], m![{ Features } % 4]>()
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inverse)
                .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &gain),
        }
    }
}
