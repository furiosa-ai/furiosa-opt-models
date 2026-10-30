use super::*;
use crate::Op;
use crate::ops::vector::Quantization;

/// The copied rows' first wave as E4M3 rows and their BF16 row scales.
pub(super) struct First;

/// The copied rows and the quantized rows and scales a quantization writes.
type Operands<'a, Hidden, QWidth, Capacity> = (
    &'a InDm<Hidden, QWidth, Capacity>,
    (
        &'a mut Source<Hidden, QWidth, Capacity>,
        &'a mut ReplicatedScales<Hidden, QWidth, Capacity>,
    ),
);

/// The quantization of the first wave of each chunk's quarter of the copied rows.
type Tile<Hidden, QWidth, Capacity> = Quantization<
    m![Capacity % 32, QWidth % 128 / 8],
    m![Capacity % 32, QWidth % 128 / 4],
    QWidth,
    m![Capacity % 32],
    m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
    m![Capacity % 32 / 4],
    m![Capacity % 4],
>;

/// The quantization of each chunk's quarter of the copied rows.
type Quarter<Hidden, QWidth, Capacity> = Quantization<
    m![Capacity % (Capacity / 4), QWidth % 128 / 8],
    m![Capacity % (Capacity / 4), QWidth % 128 / 4],
    QWidth,
    m![Capacity % (Capacity / 4)],
    m![Hidden / 1024, Capacity / (Capacity / 4), Replicas],
    m![Capacity % (Capacity / 4) / 4],
    m![Capacity % 4],
>;

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<Operands<'a, Hidden, QWidth, Capacity>> for First
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        (input, (quantized, scales)): Operands<'a, Hidden, QWidth, Capacity>,
    ) {
        let window = || {
            input.view().tile::<m![Capacity % (Capacity / 4) / 32], 1,
            m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32, QWidth % 128]>(0)
        };
        Tile::<Hidden, QWidth, Capacity>::scales(&mut context.main, window())
            .commit_view(scales.view_mut().tile::<m![Capacity % (Capacity / 4) / 32], 1,
                    m![Capacity % (Capacity / 4) / 32 = 1 #{!} 4, Capacity % 32]>(0));
        // SAFETY: the reduction broadcasts each row's scale to every chunk's slice of its quarter.
        let broadcast = unsafe {
            scales
                .view()
                .reshape::<
                    Chip,
                    Cluster<QWidth>,
                    RowWise<Hidden, QWidth, Capacity>,
                    m![Capacity % (Capacity / 4)],
                >()
        };
        let scale: VrfTensor<f32, Chip, Cluster<QWidth>, RowWise<Hidden, QWidth, Capacity>,
            m![Capacity % 32]> =
            context.sub.begin(broadcast.tile::<m![Capacity % (Capacity / 4) / 32], 1,
                m![Capacity % (Capacity / 4) / 32 = 1 # 4, Capacity % 32]>(0))
            .fetch::<m![Capacity % 32 / 8], m![Capacity % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Capacity % 32 / 8], m![Capacity % 8]>()
            .to_vrf();
        Tile::<Hidden, QWidth, Capacity>::rows(&mut context.main, window(), &scale)
            .commit_view(quantized.view_mut().tile::<m![Capacity % (Capacity / 4) / 32], 1,
                    m![Capacity % (Capacity / 4) / 32 = 1 #{!} 4, Capacity % 32,
                        QWidth % 128]>(0));
    }
}

/// The first `waves` waves of the copied rows as E4M3 rows and their BF16 row scales.
pub(super) struct All;

impl<'a, Hidden: AxisName, QWidth: AxisName, Capacity: AxisName>
    Op<(Operands<'a, Hidden, QWidth, Capacity>, usize)> for All
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        ((input, (quantized, scales)), waves): (Operands<'a, Hidden, QWidth, Capacity>, usize),
    ) {
        Quarter::<Hidden, QWidth, Capacity>::scales(&mut context.main, input.view())
            .commit_prefix::<m![Capacity % (Capacity / 4) / 4], _, _>(scales.view_mut(), 2 * waves);
        // SAFETY: the reduction broadcasts each row's scale to every chunk's slice of its quarter.
        let broadcast = unsafe {
            scales
                .view()
                .reshape::<
                    Chip,
                    Cluster<QWidth>,
                    RowWise<Hidden, QWidth, Capacity>,
                    m![Capacity % (Capacity / 4)],
                >()
        };
        let scale: VrfTensor<
            f32,
            Chip,
            Cluster<QWidth>,
            RowWise<Hidden, QWidth, Capacity>,
            m![Capacity % (Capacity / 4)],
        > = context
            .sub
            .begin(broadcast)
            .fetch::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
            .fetch_cast::<f32>()
            .collect::<m![Capacity % (Capacity / 4) / 8], m![Capacity % 8]>()
            .to_vrf();
        Quarter::<Hidden, QWidth, Capacity>::rows(&mut context.main, input.view(), &scale)
            .commit_prefix::<m![Capacity % (Capacity / 4)], _, _>(quantized.view_mut(), 8 * waves);
    }
}
