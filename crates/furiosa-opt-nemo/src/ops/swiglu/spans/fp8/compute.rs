use super::*;

fn sweep<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    input: &Input<Hidden, Intermediate, Rows>,
    weight: &Panel<Hidden, Intermediate>,
    (output, term): (&mut Both<Intermediate, Rows>, usize),
    (panel, first): (usize, Lanes<Hidden, Intermediate, Rows>),
    (live, rows): (usize, usize),
) -> Lanes<Hidden, Intermediate, Rows> {
    let mut current = first;
    for wave in 0..live {
        let next = (wave + 5 - live) / 4;
        let filled =
            lanes::<Hidden, Intermediate, Rows>(context, input, (panel, wave + 1 - next * live));
        contract::<Hidden, Intermediate, Rows>(
            context,
            (weight, &current),
            (output, term, wave),
            rows,
        );
        current = filled;
    }
    current
}

fn sweep_both<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    input: &Input<Hidden, Intermediate, Rows>,
    weight: &GateUp<Hidden, Intermediate>,
    output: &mut Both<Intermediate, Rows>,
    (panel, first): (usize, Lanes<Hidden, Intermediate, Rows>),
    (live, rows): (usize, usize),
) -> Lanes<Hidden, Intermediate, Rows> {
    let mut current = first;
    for wave in 0..live {
        let next = (wave + 5 - live) / 4;
        let filled = lanes::<Hidden, Intermediate, Rows>(
            context,
            input,
            (panel + next, wave + 1 - next * live),
        );
        contract_both::<Hidden, Intermediate, Rows>(
            context,
            (weight, &current),
            (output, wave),
            rows,
        );
        current = filled;
    }
    current
}

fn compute<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    mlp: &Mlp8<Hidden, Intermediate>,
    (input, l, (first, second)): (
        &Input<Hidden, Intermediate, Rows>,
        Lanes<Hidden, Intermediate, Rows>,
        (&Ahead<Hidden, Intermediate>, GateUp<Hidden, Intermediate>),
    ),
    (scales, channels): (
        &RowScale<Intermediate, Rows>,
        (&Channels<Intermediate, Rows>, &Channels<Intermediate, Rows>),
    ),
    waves: (usize, usize),
) -> (
    (
        Projection<Intermediate, Rows>,
        Projection<Intermediate, Rows>,
    ),
    (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>),
)
where
    super::super::super::Down<Intermediate>: AxisName,
{
    let mut b0: Both<Intermediate, Rows> = DmTensor::new();
    let l = sweep_both::<Hidden, Intermediate, Rows>(context, input, first, &mut b0, (0, l), waves);
    let third = mlp.pair(context, 2);
    let mut b1: Both<Intermediate, Rows> = DmTensor::new();
    let l =
        sweep_both::<Hidden, Intermediate, Rows>(context, input, &second, &mut b1, (1, l), waves);
    let (g3, u3) = (mlp.panel(context, 0, 3), mlp.panel(context, 1, 3));
    let sums = add_both::<Intermediate, Rows>(context, (&b0, &b1));
    let mut b2: Both<Intermediate, Rows> = DmTensor::new();
    let l =
        sweep_both::<Hidden, Intermediate, Rows>(context, input, &third, &mut b2, (2, l), waves);
    let down1 = mlp.down(context, 1);
    let sums = add_both::<Intermediate, Rows>(context, (&sums, &b2));
    let mut b3: Both<Intermediate, Rows> = DmTensor::new();
    let l = sweep::<Hidden, Intermediate, Rows>(context, input, &g3, (&mut b3, 0), (3, l), waves);
    let gate = channel_scale::<Intermediate, Rows>(context, channels.0);
    let g = scaled::<Intermediate, Rows>(context, (&sums, &b3, 0), (scales, &gate), waves.1);
    let down2 = mlp.down(context, 2);
    sweep::<Hidden, Intermediate, Rows>(context, input, &u3, (&mut b3, 1), (3, l), waves);
    let up = channel_scale::<Intermediate, Rows>(context, channels.1);
    let up = scaled::<Intermediate, Rows>(context, (&sums, &b3, 1), (scales, &up), waves.1);
    ((g, up), (down1, down2))
}

pub(super) fn gated<Hidden: AxisName, Intermediate: AxisName, Rows: AxisName>(
    context: &mut Device,
    mlp: &Mlp8<Hidden, Intermediate>,
    normalized: &Owned<Hidden, Rows>,
    (first, channels): (
        &Ahead<Hidden, Intermediate>,
        (&Channels<Intermediate, Rows>, &Channels<Intermediate, Rows>),
    ),
    (live, rows): (usize, usize),
) -> (
    (
        Projection<Intermediate, Rows>,
        Projection<Intermediate, Rows>,
    ),
    (Down<Hidden, Intermediate>, Down<Hidden, Intermediate>),
)
where
    super::super::super::Down<Intermediate>: AxisName,
{
    let packed = gate::pack::<Rows, Hidden, _, _, _, _, _, _>(context, normalized.view());
    let stored = gate::store::<Hidden, Rows, _>(context, &packed, (live, rows));
    let scales = input_scales::<Hidden, Rows>(context, normalized);
    let scales = row_scales::<Intermediate, Rows>(context, &scales);
    let input = waves::<Hidden, Intermediate, Rows>(context, &stored, (live, rows));
    let second = mlp.pair(context, 1);
    let scale = row_scale::<Intermediate, Rows, Slice<Intermediate, Rows>>(context, scales.view());
    // SAFETY: both partitions copy rows over the outer digit and keep rows in the inner digit.
    let scales = unsafe {
        scales.reshape::<Chip, Cluster<Intermediate>,
            m![Hidden % 1024 / 32, Rows / (Rows / 8)], m![Rows % (Rows / 8)]>()
    };
    let waves = [
        wave_scale::<Hidden, Intermediate, Rows>(context, &scales, 0),
        wave_scale::<Hidden, Intermediate, Rows>(context, &scales, 1),
        wave_scale::<Hidden, Intermediate, Rows>(context, &scales, 2),
        wave_scale::<Hidden, Intermediate, Rows>(context, &scales, 3),
    ];
    let mut first_wave = Input::<Hidden, Intermediate, Rows>::new();
    quantize::<Hidden, Intermediate, Rows>(context, &input[0], (&waves[0], 0), &mut first_wave);
    let l = lanes::<Hidden, Intermediate, Rows>(context, &first_wave, (0, 0));
    let mut quantized = Input::<Hidden, Intermediate, Rows>::new();
    #[unroll]
    for wave in 0..4 {
        quantize::<Hidden, Intermediate, Rows>(
            context,
            &input[wave],
            (&waves[wave], wave),
            &mut quantized,
        );
    }
    compute::<Hidden, Intermediate, Rows>(
        context,
        mlp,
        (&quantized, l, (first, second)),
        (&scale, channels),
        (live, rows),
    )
}
