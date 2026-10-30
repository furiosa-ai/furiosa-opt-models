use super::*;
use crate::Op;
use crate::tensor::axis;
use crate::tensor::live::Live;
use core::marker::PhantomData;

/// A panel's contraction over the live waves, each wave's TRF fill one wave ahead.
pub(super) struct Sweep;

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Input<Hidden, Intermediate, Rows>,
        &'a Panel<Hidden, Intermediate>,
        (&'a mut Both<Intermediate, Rows>, usize),
        (usize, Lanes<Hidden, Intermediate, Rows>),
        (usize, usize),
    )> for Sweep
{
    type Output = Lanes<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (input, weight, (output, term), (panel, first), (live, rows)): (
            &'a Input<Hidden, Intermediate, Rows>,
            &'a Panel<Hidden, Intermediate>,
            (&'a mut Both<Intermediate, Rows>, usize),
            (usize, Lanes<Hidden, Intermediate, Rows>),
            (usize, usize),
        ),
    ) -> Self::Output {
        let mut current = first;
        for wave in 0..live {
            let next = (wave + 5 - live) / 4;
            let filled = Fill.forward(&mut *context, (input, (panel, wave + 1 - next * live)));
            Contract.forward(
                &mut *context,
                ((weight, &current), (&mut *output, term, wave), rows),
            );
            current = filled;
        }
        current
    }
}

impl<'a, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>
    Op<(
        &'a Input<Hidden, Intermediate, Rows>,
        &'a GateUp<Hidden, Intermediate>,
        &'a mut Both<Intermediate, Rows>,
        (usize, Lanes<Hidden, Intermediate, Rows>),
        (usize, usize),
    )> for Sweep
{
    type Output = Lanes<Hidden, Intermediate, Rows>;

    fn forward(
        &self,
        context: &mut Device,
        (input, weight, output, (panel, first), (live, rows)): (
            &'a Input<Hidden, Intermediate, Rows>,
            &'a GateUp<Hidden, Intermediate>,
            &'a mut Both<Intermediate, Rows>,
            (usize, Lanes<Hidden, Intermediate, Rows>),
            (usize, usize),
        ),
    ) -> Self::Output {
        let mut current = first;
        for wave in 0..live {
            let next = (wave + 5 - live) / 4;
            let filled = Fill.forward(
                &mut *context,
                (input, (panel + next, wave + 1 - next * live)),
            );
            Contract.forward(
                &mut *context,
                ((weight, &current), (&mut *output, wave), rows),
            );
            current = filled;
        }
        current
    }
}

/// The gate and up projections of a span's live rows: its rows as E4M3 by their row scales, each
/// pair of 1024-input weight panels contracted with them in turn, the next pair landing during the
/// current one's contraction, and the channel scales applied to the summed panels. A pair's
/// partial sums are added as soon as the next pair's are, so at most two of them are held.
impl<'t, Hidden: AxisName, Intermediate: AxisName, Rows: AxisName, Capacity: AxisName>
    Op<(
        GateUp<Hidden, Intermediate>,
        Live<&'t Owned<Hidden, Rows>, Capacity>,
    )> for GateUpWeights<Hidden, Intermediate>
{
    type Output = (
        Projection<Intermediate, Rows>,
        Projection<Intermediate, Rows>,
    );

    fn forward(
        &self,
        context: &mut Device,
        (first, normalized): (
            GateUp<Hidden, Intermediate>,
            Live<&'t Owned<Hidden, Rows>, Capacity>,
        ),
    ) -> Self::Output {
        const { assert!(Hidden::SIZE / 1024 == 4, "written for four passes") };
        let waves = (normalized.count::<axis::Rows<128>>(), normalized.rows());
        let (live, rows) = waves;
        let normalized = *normalized.value();
        let packed =
            gate::Pack::<Rows, Hidden, _, _>(PhantomData).forward(&mut *context, normalized.view());
        let stored = gate::Store.forward(&mut *context, (&packed, (live, rows)));
        let scales = InputScales.forward(&mut *context, normalized);
        let scales: DmTensor<
            bf16,
            Chip,
            Cluster<Intermediate>,
            Slice<Intermediate, Rows>,
            m![Rows % (Rows / 8)],
        > = scales.to_dm(&mut context.tdma);
        let input =
            Tiles::<Intermediate>(PhantomData).forward(&mut *context, (&stored, (live, rows)));
        let pair = |context: &mut Device, panel: usize| -> GateUp<Hidden, Intermediate> {
            self.forward(context, Begin(panel))
        };
        let second = pair(context, 1);
        let scale = FetchRows.forward(&mut *context, scales.view());
        // SAFETY: both partitions copy rows over the outer digit and keep rows in the inner digit.
        let scales = unsafe {
            scales.reshape::<Chip, Cluster<Intermediate>,
                m![Hidden % 1024 / 32, Rows / (Rows / 8)], m![Rows % (Rows / 8)]>()
        };
        let tiles = [
            FetchTile.forward(&mut *context, (&scales, 0)),
            FetchTile.forward(&mut *context, (&scales, 1)),
            FetchTile.forward(&mut *context, (&scales, 2)),
            FetchTile.forward(&mut *context, (&scales, 3)),
        ];
        let mut first_tile = Input::<Hidden, Intermediate, Rows>::new();
        Quantize.forward(&mut *context, (&input[0], (&tiles[0], 0), &mut first_tile));
        let l = Fill.forward(&mut *context, (&first_tile, (0, 0)));
        let mut quantized = Input::<Hidden, Intermediate, Rows>::new();
        #[unroll]
        for wave in 0..4 {
            Quantize.forward(
                &mut *context,
                (&input[wave], (&tiles[wave], wave), &mut quantized),
            );
        }
        let channels = |context: &mut Device, term: usize| -> ChannelScale<Intermediate, Rows> {
            let channels: Channels<Intermediate, Rows> = self
                .1
                .view()
                .tile::<m![Term], 1, m![Term = 1 # 2, Intermediate]>(term)
                .to_dm(&mut context.tdma);
            FetchChannels.forward(&mut *context, &channels)
        };
        let panel = |context: &mut Device, term: usize| -> Panel<Hidden, Intermediate> {
            self.0
                .view()
                .tile::<m![Term], 1, m![Term = 1 # 2, Hidden / 1024,
                    Intermediate / (Intermediate / 64), Hidden % 1024 / 128,
                    { Out<Intermediate> }, Hidden % 128]>(term)
                .tile::<m![Hidden / 1024], 1, m![Term = 1 # 2,
                    Hidden / 1024 = 1 # (Hidden / 1024), Intermediate / (Intermediate / 64),
                    Hidden % 1024 / 128, { Out<Intermediate> }, Hidden % 128]>(
                    const { Hidden::SIZE / 1024 - 1 },
                )
                .to_dm(&mut context.tdma)
        };
        let add = |context: &mut Device,
                   (left, right): (&Both<Intermediate, Rows>, &Both<Intermediate, Rows>)|
         -> Both<Intermediate, Rows> {
            let mut sums: Both<Intermediate, Rows> = DmTensor::new();
            Add::of::<
                m![Rows % (Rows / 8), Term],
                m![{ Out<Intermediate> } / 8],
                m![{ Out<Intermediate> } % 8],
                Side,
                _,
                _,
                _,
                _,
            >(&mut *context, (left, right))
            .commit_view(sums.view_mut());
            sums
        };
        let mut b0: Both<Intermediate, Rows> = DmTensor::new();
        let l = Sweep.forward(&mut *context, (&quantized, &first, &mut b0, (0, l), waves));
        let third = pair(context, 2);
        let mut b1: Both<Intermediate, Rows> = DmTensor::new();
        let l = Sweep.forward(&mut *context, (&quantized, &second, &mut b1, (1, l), waves));
        let sums = add(context, (&b0, &b1));
        let (g3, u3) = (panel(context, 0), panel(context, 1));
        let mut b2: Both<Intermediate, Rows> = DmTensor::new();
        let l = Sweep.forward(&mut *context, (&quantized, &third, &mut b2, (2, l), waves));
        let sums = add(context, (&sums, &b2));
        let mut b3: Both<Intermediate, Rows> = DmTensor::new();
        let l = Sweep.forward(
            &mut *context,
            (&quantized, &g3, (&mut b3, 0), (3, l), waves),
        );
        let channel = channels(context, 0);
        let gate = Combine.forward(&mut *context, ((&sums, &b3, 0), (&scale, &channel), rows));
        Sweep.forward(
            &mut *context,
            (&quantized, &u3, (&mut b3, 1), (3, l), waves),
        );
        let channel = channels(context, 1);
        let up = Combine.forward(&mut *context, ((&sums, &b3, 1), (&scale, &channel), rows));
        (gate, up)
    }
}
