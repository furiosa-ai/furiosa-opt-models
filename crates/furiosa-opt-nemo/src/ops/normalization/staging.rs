//! Consumed FP32 statistics keep the norm and input bound until the consumer takes the stream.

use furiosa_opt_std::prelude::*;

use super::RmsNorm;
use crate::compose::{Once, Space};
use crate::ops::layout::Dm;

mod completion;
mod replicated;

type Sums<C, Cl, S, R> = DmTensor<f32, C, Cl, S, m![{ R }, 1 # 8]>;

pub struct Partial<V>(V);
pub struct Gathered<V>(V);
pub struct Reduced<V, Level>(V, core::marker::PhantomData<Level>);
pub struct Exchange<V, Level>(V, V, core::marker::PhantomData<Level>);
pub struct Total<V>(V);
pub struct Local;
pub struct Cluster;
pub struct Chip;

impl<'a, W: AxisName, const E: usize, C: M, Shape: M>
    Once<&'a RmsNorm<W, E, C, crate::ops::weight::Weight<Shape, C>>>
{
    /// Consume this gain binding; callers must not recreate it for the same layer in a launch.
    pub fn stage<Cl: M, S: M, L: M>(
        self,
        device: &mut Device,
    ) -> RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>> {
        self.read(|norm| norm).stage(device)
    }
}

/// The statistic carries the exact norm and residual that produced it, not a caller-chosen pair.
pub struct Statistics<'a, N, I, V> {
    norm: &'a N,
    input: &'a I,
    value: V,
}

impl<W: AxisName, const E: usize, C: M, Cl: M, S: M, L: M>
    RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>>
{
    pub fn statistics<'a, T: Space, R: M, Groups: M, Features: M, Input: M>(
        &'a self,
        device: &mut Device,
        input: &'a Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
    ) -> Statistics<
        'a,
        Self,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Partial<Sums<C, Cl, S, R>>,
    > {
        const {
            assert!(W::SIZE.is_multiple_of(L::SIZE));
            assert!(L::SIZE == Groups::SIZE * Features::SIZE);
            assert!(Features::SIZE.is_multiple_of(8));
        }
        Statistics {
            norm: self,
            input,
            value: Partial(
                device
                    .main
                    .begin(input.value.clone())
                    .fetch::<m![{ R }, { Groups }, { Features } / 8], m![{ Features } % 8]>()
                    .fetch_cast::<f32>()
                    .collect::<m![{ R }, { Groups }, { Features } / 8], m![{ Features } % 8]>()
                    .vector_init()
                    .vector_intra_slice_tag(TagMode::Zero)
                    .vector_narrow_split::<m![{ R }, { Groups }, { Features } / 4], m![{ Features } % 4]>()
                    .vector_stash()
                    .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
                    .vector_intra_slice_reduce::<W, R, m![1 # 4]>(IntraSliceReduceOpF32::Add)
                    .vector_fp_div(const { W::SIZE as f32 })
                    .vector_widen_pad::<m![1 # 8]>()
                    .vector_clip(
                        ClipBinaryOpF32::Add,
                        const { f32::from_bits(E as u32) / (W::SIZE / L::SIZE) as f32 },
                    )
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit(),
            ),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M> Statistics<'a, N, I, Partial<Sums<C, Cl, S, R>>> {
    /// The mappings must cover each row's feature partitions once; counts alone do not prove this.
    pub fn gather<Copies: M, Features: M>(
        self,
        device: &mut Device,
    ) -> Statistics<'a, N, I, Gathered<DmTensor<f32, C, Cl, Copies, m![{ Features }, { R }, 1 # 8]>>>
    {
        const {
            assert!(S::SIZE.is_multiple_of(Features::SIZE));
            assert!(Copies::SIZE.is_multiple_of(S::SIZE / Features::SIZE));
        }
        let mut gathered = DmTensor::new();
        self.value
            .0
            .view()
            .to_dm_view(&mut device.tdma, gathered.view_mut());
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Gathered(gathered),
        }
    }
}

impl<
    'a,
    W: AxisName,
    const E: usize,
    C: M,
    Cl: M,
    S: M,
    L: M,
    T: Space,
    F: M,
    R: M,
    Copies: M,
    Input: M,
>
    Statistics<
        'a,
        RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>>,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Gathered<DmTensor<f32, C, Cl, Copies, m![{ F }, { R }, 1 # 8]>>,
    >
{
    pub fn reduce(
        self,
        device: &mut Device,
    ) -> Statistics<
        'a,
        RmsNorm<W, E, C, DmTensor<bf16, C, Cl, S, L>>,
        Dm<T, DmTensorView<'a, bf16, C, Cl, S, Input>>,
        Reduced<Sums<C, Cl, S, R>, Local>,
    > {
        const {
            assert!(W::SIZE == C::SIZE * Cl::SIZE * F::SIZE * L::SIZE);
            assert!(Copies::SIZE == S::SIZE);
        }
        let sums: Sums<C, Cl, Copies, R> = device
            .main
            .begin(self.value.0.view())
            .fetch::<m![{ F }, { R }], m![1 # 8]>()
            .collect::<m![{ F }, { R }], m![1 # 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_trim::<m![1 # 4]>()
            .vector_intra_slice_reduce::<W, R, m![1 # 4]>(IntraSliceReduceOpF32::Add)
            .vector_widen_pad::<m![1 # 8]>()
            .vector_final()
            .commit_trim::<m![1 # 8]>()
            .commit();
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Reduced(
                // SAFETY: gather copies keep the original row-slice order; only feature-copy labels change.
                unsafe { sums.reshape() },
                core::marker::PhantomData,
            ),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M> Statistics<'a, N, I, Reduced<Sums<C, Cl, S, R>, Local>> {
    pub fn clusters(
        self,
        device: &mut Device,
    ) -> Statistics<'a, N, I, Exchange<Sums<C, Cl, S, R>, Cluster>> {
        const {
            assert!(Cl::SIZE == 2);
        }
        let peer = self.value.0.cluster_swap().to_dm(&mut device.tdma);
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Exchange(self.value.0, peer, core::marker::PhantomData),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M> Statistics<'a, N, I, Reduced<Sums<C, Cl, S, R>, Cluster>> {
    pub fn chips(
        self,
        device: &mut Device,
    ) -> Statistics<'a, N, I, Exchange<Sums<C, Cl, S, R>, Chip>> {
        const {
            assert!(C::SIZE == 2);
        }
        let peer = self.value.0.chip_shuffle([1, 0]).to_dm(&mut device.tdma);
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Exchange(self.value.0, peer, core::marker::PhantomData),
        }
    }

    pub fn finish(self) -> Statistics<'a, N, I, Total<Sums<C, Cl, S, R>>> {
        const {
            assert!(C::SIZE == 1);
        }
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Total(self.value.0),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M> Statistics<'a, N, I, Reduced<Sums<C, Cl, S, R>, Chip>> {
    pub fn finish(self) -> Statistics<'a, N, I, Total<Sums<C, Cl, S, R>>> {
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Total(self.value.0),
        }
    }
}

impl<'a, N, I, C: M, Cl: M, S: M, R: M, Level>
    Statistics<'a, N, I, Exchange<Sums<C, Cl, S, R>, Level>>
{
    pub fn combine(
        self,
        device: &mut Device,
    ) -> Statistics<'a, N, I, Reduced<Sums<C, Cl, S, R>, Level>> {
        axes![Term = 2];
        Statistics {
            norm: self.norm,
            input: self.input,
            value: Reduced(
                device
                    .main
                    .begin_interleaved::<Term, _, _, _, _, _>(
                        self.value.0.view(),
                        self.value.1.view(),
                    )
                    .fetch::<m![{ R }, Term], m![1 # 8]>()
                    .collect::<m![{ R }, Term], m![1 # 8]>()
                    .vector_init()
                    .vector_intra_slice_unzip::<Term, R>()
                    .vector_clip_zip(ClipBinaryOpF32::Add)
                    .vector_final()
                    .commit_trim::<m![1 # 8]>()
                    .commit(),
                core::marker::PhantomData,
            ),
        }
    }
}
