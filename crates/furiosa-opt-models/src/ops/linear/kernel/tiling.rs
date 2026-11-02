use super::*;

pub(crate) trait Tiling<In: AxisName, Out: AxisName>: 'static {
    type Cluster: M;
    type Slices: M;
    type Rows: M;
    type Inputs: M;
    type Restorations: prelude::Count;
    type Restored: M;

    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: HbmTensorView<'_, D, X, m![Out, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Self::Cluster, Self::Slices, E>;
}

impl<In: AxisName, Out: AxisName> Contraction<In, Out> {
    const TILED: () = {
        let c = Self::config();
        assert!(contraction::legal(
            In::SIZE,
            Out::SIZE,
            (c.chunk, c.block, c.passes)
        ));
    };
}

impl<In: AxisName, Out: AxisName> Tiling<In, Out> for Contraction<In, Out> {
    type Cluster = Cluster<In, Out>;
    type Slices = ProjectionSlices<In, Out>;
    type Rows = m![Out % ({ Block<In, Out> })];
    type Inputs = Chunked<In, Out>;
    type Restorations = Restorations<In, Out>;
    type Restored = m![Out % ({ Symbol<RestoredRows<In, Out>> })];

    fn to_dm<D: MaterializableScalar, X: M, K: M, E: M>(
        device: &mut Device,
        tensor: HbmTensorView<'_, D, X, m![Out, { K }]>,
        at: usize,
    ) -> DmTensor<D, X, Cluster<In, Out>, ProjectionSlices<In, Out>, E> {
        const { Self::TILED };
        tensor
            .tile::<SlabIndex<In, Out>, 1, m![Out % (Out / ({ Passes<In, Out> })) # Out, { K }]>(at)
            .to_dm(&mut device.tdma)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RestoredRows<In, Out>(PhantomData<(In, Out)>);

impl<In: AxisName, Out: AxisName> AxisName for RestoredRows<In, Out> {
    const NAME: Ident = Ident::new("RestoredRows");
    const SIZE: usize = contraction::restored(Contraction::<In, Out>::config().block);
}

type Restorations<In, Out> = m![Out % ({ Block<In, Out> }) / ({ Symbol<RestoredRows<In, Out>> })];
