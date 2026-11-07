//! A step's operands, and the one preparation of their context that a catalog call borrows.

use super::{Count, Space, Tiles};
use crate::Module;
use crate::modules::language::Indexed;
use crate::ops::layout::{Packets, axes::Chip};
use furiosa_opt_std::prelude::*;

axes![Fields = 5];
pub const FIELDS: usize = Fields::SIZE;

/// Logical history operands, a copy on each chip of `Chips`; extents name segments, stored blocks
/// and selected rows, not live rows.
#[derive(DeviceSend)]
pub struct History<Segments: AxisName, Blocks: AxisName, Selected: AxisName, Chips: M = Chip> {
    pub segments: HbmTensor<i32, Chips, m![Segments, Fields]>,
    pub blocks: HbmTensor<i32, Chips, m![Blocks]>,
    pub selected: HbmTensor<i32, Chips, m![Selected]>,
    pub extents: HbmTensor<i32, Chips, m![3]>,
}

/// Device operands of one packed prefix on the chips `Chips`; cache includes its logical history
/// operands.
#[derive(DeviceSend)]
pub struct Input<S, Tokens, Cache, Chips: M = Chip> {
    tokens: Tokens,
    count: Count<Chips>,
    cache: Cache,
    space: std::marker::PhantomData<S>,
}

impl<S, T, C, Chips: M> Input<S, T, C, Chips> {
    pub const fn new(tokens: T, count: Count<Chips>, cache: C) -> Self {
        Self {
            tokens,
            count,
            cache,
            space: std::marker::PhantomData,
        }
    }
}

/// How a call's cache operands reach the catalog's layers with the contexts its step prepared.
pub trait Lend<Prepared> {
    type Context;

    fn lend(self, prepared: Prepared) -> Self::Context;
}

/// What a step lends its body: the call's cache operands with the contexts the step prepared.
pub type Lent<'c, 'p, Cache, S, C> = <&'c mut Cache as Lend<&'p Prepared<S, C>>>::Context;

/// Prepared attention geometries share the only live domain of the invocation.
pub struct Prepared<S, Context> {
    tiles: Tiles<S>,
    context: Context,
}

impl<S: Space, C> Prepared<S, C> {
    pub const fn tiles(&self) -> Tiles<S> {
        self.tiles
    }

    pub const fn context(&self) -> &C {
        &self.context
    }
}

/// A step reads its live rows, `index` prepares their context from its cache operands once, and
/// the tokens enter `body` with both; the context cannot escape the call.
impl<S: Space, T, Cache, Chips: M, C, O, Index, Body>
    Module<(&mut Device, &mut Input<S, T, Cache, Chips>)> for Indexed<Index, Body>
where
    Index: for<'d, 'c> Module<(&'d mut Device, (Tiles<S>, &'c mut Cache)), Output = C>,
    for<'c, 'p> &'c mut Cache: Lend<&'p Prepared<S, C>>,
    Body: for<'d, 't, 'c, 'p> Module<
            (
                &'d mut Device,
                (&'t T, (Tiles<S>, Lent<'c, 'p, Cache, S, C>)),
            ),
            Output = O,
        >,
{
    type Output = O;

    fn forward(&self, (device, input): (&mut Device, &mut Input<S, T, Cache, Chips>)) -> O {
        let tiles = Tiles::read(&input.count);
        let prepared = Prepared {
            tiles,
            context: self
                .index
                .forward((&mut *device, (tiles, &mut input.cache))),
        };
        let lent = (&mut input.cache).lend(&prepared);
        self.body.forward((device, (&input.tokens, (tiles, lent))))
    }
}

/// A call's tokens as its catalog takes them, beside the context the step lends the catalog.
#[derive(DeviceSend)]
pub struct Enter;

/// One token ID enters through the catalog's embedding.
impl<'t, D, S, C, E: Scalar, P: M> Module<(&mut D, (&'t HbmTensor<E, P, m![1]>, (Tiles<S>, C)))>
    for Enter
{
    type Output = (&'t HbmTensor<E, P, m![1]>, C);

    fn forward(
        &self,
        (_, (token, (_, context))): (&mut D, (&'t HbmTensor<E, P, m![1]>, (Tiles<S>, C))),
    ) -> Self::Output {
        (token, context)
    }
}

/// A launch's token IDs enter through the catalog's embedding with the step's live rows.
impl<'t, D, S, C, P: M, R: AxisName>
    Module<(&mut D, (&'t HbmTensor<i32, P, Packets<R>>, (Tiles<S>, C)))> for Enter
{
    type Output = ((&'t HbmTensor<i32, P, Packets<R>>, Tiles<S>), C);

    fn forward(
        &self,
        (_, (tokens, (tiles, context))): (
            &mut D,
            (&'t HbmTensor<i32, P, Packets<R>>, (Tiles<S>, C)),
        ),
    ) -> Self::Output {
        ((tokens, tiles), context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sequential;
    use crate::compose::tests::count;
    use furiosa_opt_std::runtime::launch;

    axes![Two = 2, Eight = 8];
    impl Space for Two {
        type Envelope = Eight;
    }

    type Token = HbmTensor<i32, Chip, m![1]>;

    impl<'c, 'p, S, C> Lend<&'p Prepared<S, C>> for &'c mut Token {
        type Context = (Self, &'p Prepared<S, C>);

        fn lend(self, prepared: &'p Prepared<S, C>) -> Self::Context {
            (self, prepared)
        }
    }

    /// A geometry that is a copy of the cache operand.
    #[derive(DeviceSend)]
    struct Geometry;
    impl Module<(&mut Device, (Tiles<Two>, &mut Token))> for Geometry {
        type Output = Token;

        fn forward(&self, (device, (_, cache)): (&mut Device, (Tiles<Two>, &mut Token))) -> Token {
            cache.to_hbm(&mut device.tdma)
        }
    }

    /// A catalog that writes its token into the cache operand and returns the geometry it borrows.
    #[derive(DeviceSend)]
    struct Catalog;
    impl<'t, 'c, 'p>
        Module<(
            &mut Device,
            (&'t Token, (&'c mut Token, &'p Prepared<Two, Token>)),
        )> for Catalog
    {
        type Output = Token;

        fn forward(
            &self,
            (device, (token, (cache, prepared))): (
                &mut Device,
                (&'t Token, (&'c mut Token, &'p Prepared<Two, Token>)),
            ),
        ) -> Token {
            token.view().to_hbm_view(&mut device.tdma, cache.view_mut());
            prepared.context().to_hbm(&mut device.tdma)
        }
    }

    type Step = Indexed<Geometry, Sequential<Enter, Catalog>>;

    /// The shared entry takes a step's whole input on either backend.
    #[test]
    fn launches_on_the_entry() {
        fn accepts<F>(_: F)
        where
            F: Loadable<
                    (
                        &'static mut Device,
                        &'static Step,
                        &'static mut Input<Two, Token, Token>,
                    ),
                    Output = Token,
                >,
        {
        }
        accepts(crate::models::forward);
    }

    /// The catalog borrows the context prepared from the cache operands before it writes them.
    #[tokio::test(flavor = "current_thread")]
    async fn prepares_before_the_catalog() {
        let _lease = crate::tests::DEVICE.lock().await;
        let mut device = Device::new(crate::models::forward.topology()).unwrap();
        let tokens: Token = HostTensor::<i32, m![1]>::from_vec(vec![5])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let cache: Token = HostTensor::<i32, m![1]>::from_vec(vec![7])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let live = count(1).to_hbm(&mut device.pdma).await.unwrap();
        let mut input = Input::<Two, _, _>::new(tokens, live, cache);
        let step: Step = Indexed {
            index: Geometry,
            body: Sequential(Enter, Catalog),
        };
        let borrowed = launch(crate::models::forward, (&mut device, &step, &mut input))
            .await
            .unwrap();
        let Input { cache, .. } = input;
        let borrowed = borrowed.to_host::<m![1]>(&mut device.pdma).await.unwrap();
        let cache = cache.to_host::<m![1]>(&mut device.pdma).await.unwrap();
        assert_eq!((borrowed.into_vec(), cache.into_vec()), (vec![7], vec![5]));
    }
}
