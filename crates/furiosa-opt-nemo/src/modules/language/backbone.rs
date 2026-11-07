//! The language backbone: embed the input, then run the body with the invocation context.

use crate::compose::{Tiles, step::Enter};
use crate::ops::embedding::Reorder;
use crate::ops::layout::{Dm, Relayout};
use crate::ops::normalization::Spread;
use crate::{DeviceModule, Module};
use furiosa_opt_std::prelude::*;

/// Embeds the input, then passes hidden values and invocation context to the body.
#[derive(Clone, DeviceSend)]
pub struct Backbone<Embedding, Body> {
    pub embedding: Embedding,
    pub body: Body,
}

/// Supplied hidden rows are the complete embedding subtree's output, including its scale.
#[derive(DeviceSend)]
pub struct Embedded<X, C = ()> {
    pub hidden: X,
    pub context: C,
}

/// A supplied row enters past the embedding subtree; the body owns a copy, as it owns rows it
/// embeds.
impl<'t, S, C, D: MaterializableScalar, P: M, W: AxisName>
    Module<(
        &mut Device,
        (&'t Embedded<HbmTensor<D, P, m![W]>>, (Tiles<S>, C)),
    )> for Enter
{
    type Output = Embedded<HbmTensor<D, P, m![W]>, C>;

    fn forward(
        &self,
        (device, (rows, (_, context))): (
            &mut Device,
            (&'t Embedded<HbmTensor<D, P, m![W]>>, (Tiles<S>, C)),
        ),
    ) -> Self::Output {
        let hidden = Relayout::<m![W]>::default().forward((device, &rows.hidden));
        Embedded { hidden, context }
    }
}

/// A launch's supplied rows enter past the embedding subtree in the rows the body computes on,
/// with the step's live rows.
impl<'t, S, C, W: AxisName, R: AxisName>
    Module<(
        &mut Device,
        (
            &'t Embedded<HbmTensor<bf16, m![1], m![R, W]>>,
            (Tiles<S>, C),
        ),
    )> for Enter
{
    type Output = Embedded<Dm<S, Spread<W, R>>, C>;

    fn forward(
        &self,
        (device, (rows, (tiles, context))): (
            &mut Device,
            (
                &'t Embedded<HbmTensor<bf16, m![1], m![R, W]>>,
                (Tiles<S>, C),
            ),
        ),
    ) -> Self::Output {
        let hidden = Reorder::<W>::default().forward((device, (&rows.hidden, tiles)));
        Embedded { hidden, context }
    }
}

impl<I, C, X, Y, Embedding, Body> Module<(&mut Device, (I, C))> for Backbone<Embedding, Body>
where
    Embedding: DeviceModule<I, Output = X>,
    Body: DeviceModule<(X, C), Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, (input, context)): (&mut Device, (I, C))) -> Self::Output {
        let hidden = self.embedding.forward((&mut *device, input));
        self.body.forward((device, (hidden, context)))
    }
}

impl<'i, I, C, X, Y, Embedding, Body> Module<(&mut Device, &'i mut (I, C))>
    for Backbone<Embedding, Body>
where
    Embedding: DeviceModule<&'i I, Output = X>,
    Body: DeviceModule<(X, &'i mut C), Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, (input, context)): (&mut Device, &'i mut (I, C))) -> Y {
        self.forward((device, (&*input, context)))
    }
}

impl<K, I, C, X, Y, Embedding, Body> Module<(&mut Device, (K, I, C))> for Backbone<Embedding, Body>
where
    Embedding: DeviceModule<(K, I), Output = X>,
    Body: DeviceModule<(X, C), Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, (input, condition, context)): (&mut Device, (K, I, C))) -> Y {
        self.forward((device, ((input, condition), context)))
    }
}

impl<X, C, Y, Embedding, Body> Module<(&mut Device, Embedded<X, C>)> for Backbone<Embedding, Body>
where
    Body: DeviceModule<(X, C), Output = Y>,
{
    type Output = Y;

    fn forward(&self, (device, Embedded { hidden, context }): (&mut Device, Embedded<X, C>)) -> Y {
        self.body.forward((device, (hidden, context)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layers, Then, modules::language::Normalized};

    struct Id(usize);
    struct Hidden(i32);
    struct Embedding([i32; 2]);
    struct Scale(i32);
    struct Layer(i32);
    struct Norm(i32);

    impl Module<(&mut Device, &Id)> for Embedding {
        type Output = Hidden;

        fn forward(&self, (_, id): (&mut Device, &Id)) -> Hidden {
            Hidden(self.0[id.0])
        }
    }

    impl Module<(&mut Device, &Hidden)> for Scale {
        type Output = Hidden;

        fn forward(&self, (_, hidden): (&mut Device, &Hidden)) -> Hidden {
            Hidden(hidden.0 * self.0)
        }
    }

    impl Module<(&mut Device, (&Hidden, &mut i32))> for Layer {
        type Output = Hidden;

        fn forward(&self, (_, (hidden, effects)): (&mut Device, (&Hidden, &mut i32))) -> Hidden {
            *effects += 1;
            Hidden(hidden.0 + self.0)
        }
    }

    impl Module<(&mut Device, &Hidden)> for Norm {
        type Output = Hidden;

        fn forward(&self, (_, hidden): (&mut Device, &Hidden)) -> Hidden {
            Hidden(hidden.0 * self.0)
        }
    }

    /// IDs and post-scale rows bind to the same normalized catalog without a replacement tree.
    #[test]
    fn accepts_ids_and_embedded_rows() {
        let call = |device: &mut Device| {
            let model = Normalized {
                body: Backbone {
                    embedding: Then(Embedding([7, 13]), Scale(3)),
                    body: Layers {
                        layers: [Layer(5), Layer(11)],
                    },
                },
                norm: Norm(2),
            };
            let ids = model.forward((&mut *device, (&Id(0), &mut [0, 0])));
            let embedded = model.forward((
                device,
                Embedded {
                    hidden: Hidden(21),
                    context: &mut [0, 0],
                },
            ));
            (ids, embedded)
        };
        let _: fn(&mut Device) -> (Hidden, Hidden) = call;
    }
}
