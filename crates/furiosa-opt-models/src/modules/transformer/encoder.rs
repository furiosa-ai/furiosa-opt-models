//! An encoder stack that copies selected layer inputs out as features before its final norm.

use crate::Module;
use crate::ops::index::scatter::Store;
use furiosa_opt_std::prelude::{
    Broadcast, Device, DeviceSend, HbmTensor, M, MaterializableScalar, m, unroll,
};

/// Applies the layers and the final norm, copying selected layer inputs out as features.
#[derive(DeviceSend)]
pub struct Encoder<Embedding, Layers, Norm> {
    pub embedding: Embedding,
    pub layers: Layers,
    pub norm: Norm,
}

/// The state holds the embedding's slot, one slot a layer and the norm's; the layers' slots hold no
/// borrowed state, so each is lent by index.
impl<
    's,
    I,
    Embedding,
    Layer,
    Norm,
    EmbeddingState,
    LayerState,
    NormState,
    D: MaterializableScalar,
    S: M,
    const COUNT: usize,
    Env: Copy,
>
    Module<
        (
            I,
            &mut [HbmTensor<D, m![1], S>; COUNT],
            &mut HbmTensor<D, m![1], S>,
        ),
        Env,
        &'s mut (EmbeddingState, [LayerState; COUNT], NormState),
    > for Encoder<Embedding, [Layer; COUNT], Norm>
where
    Embedding: Module<I, Env, &'s mut EmbeddingState, Output = HbmTensor<D, m![1], S>>,
    Layer: for<'t> Module<
            HbmTensor<D, m![1], S>,
            Env,
            &'t mut LayerState,
            Output = HbmTensor<D, m![1], S>,
        >,
    Norm: Module<HbmTensor<D, m![1], S>, Env, &'s mut NormState, Output = HbmTensor<D, m![1], S>>,
{
    type Output = HbmTensor<D, m![1], S>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, layers, norm): &'s mut (EmbeddingState, [LayerState; COUNT], NormState),
        (input, features, last): (I, &mut [Self::Output; COUNT], &mut Self::Output),
    ) -> Self::Output {
        let mut hidden = self.embedding.forward(&mut *device, env, embedding, input);
        if COUNT > 0 {
            #[unroll]
            for index in 0..COUNT {
                Store.forward(&mut *device, env, &mut (), (&hidden, &mut features[index]));
                hidden = self.layers[index].forward(&mut *device, env, &mut layers[index], hidden);
            }
        }
        Store.forward(&mut *device, env, &mut (), (&hidden, last));
        self.norm.forward(device, env, norm, hidden)
    }
}

/// Capture selected layer inputs without allocating every intermediate activation.
impl<
    's,
    I,
    Embedding,
    Layer,
    Norm,
    EmbeddingState,
    LayerState,
    NormState,
    D: MaterializableScalar,
    S: M,
    const DEPTH: usize,
    const COUNT: usize,
    Env: Copy,
>
    Module<
        (I, (&[usize; COUNT], &mut [HbmTensor<D, m![1], S>; COUNT])),
        Env,
        &'s mut (EmbeddingState, [LayerState; DEPTH], NormState),
    > for Encoder<Embedding, [Layer; DEPTH], Norm>
where
    Embedding: Module<I, Env, &'s mut EmbeddingState, Output = HbmTensor<D, m![1], S>>,
    Layer: for<'t> Module<
            HbmTensor<D, m![1], S>,
            Env,
            &'t mut LayerState,
            Output = HbmTensor<D, m![1], S>,
        >,
    Norm: Module<HbmTensor<D, m![1], S>, Env, &'s mut NormState, Output = HbmTensor<D, m![1], S>>,
{
    type Output = HbmTensor<D, m![1], S>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        (embedding, layers, norm): &'s mut (EmbeddingState, [LayerState; DEPTH], NormState),
        (input, (indices, features)): (I, (&[usize; COUNT], &mut [Self::Output; COUNT])),
    ) -> Self::Output {
        let mut hidden = self.embedding.forward(&mut *device, env, embedding, input);
        if DEPTH > 0 {
            #[unroll]
            #[expect(
                clippy::needless_range_loop,
                reason = "device unrolling requires index ranges"
            )]
            for index in 0..DEPTH {
                if COUNT > 0 {
                    #[unroll]
                    for capture in 0..COUNT {
                        if indices[capture] == index {
                            Store.forward(
                                &mut *device,
                                env,
                                &mut (),
                                (&hidden, &mut features[capture]),
                            );
                        }
                    }
                }
                hidden = self.layers[index].forward(&mut *device, env, &mut layers[index], hidden);
            }
        }
        if COUNT > 0 {
            #[unroll]
            for capture in 0..COUNT {
                if indices[capture] == DEPTH {
                    Store.forward(
                        &mut *device,
                        env,
                        &mut (),
                        (&hidden, &mut features[capture]),
                    );
                }
            }
        }
        self.norm.forward(device, env, norm, hidden)
    }
}
