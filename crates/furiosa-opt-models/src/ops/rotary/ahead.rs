//! Prefetch rotary coefficient rows alongside the Q/K tiles that consume them.

use furiosa_opt_std::prelude::*;

use super::{
    CoefficientRows, Lines, OutputHalf, RotateTile, Rotation, Row, TileCoefficients, tiles,
};
use crate::Op;
use crate::tensor::live::Live;

type LandedTable<Width, Head, Chips> = DmTensor<
    bf16,
    Chips,
    tiles::Half<Width, Head>,
    tiles::Slices<Width, Head>,
    m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> }],
>;

pub(super) fn write_live<Head: AxisName, Width: AxisName, R: AxisName, Chips: M>(
    device: &mut Device,
    (input, rotation): (
        Live<
            (
                &HbmTensor<bf16, Chips, m![R, Width]>,
                &HbmTensor<bf16, Chips, m![R, Width]>,
            ),
            R,
        >,
        &Rotation<Head, Chips, R>,
    ),
    output: (
        &mut HbmTensor<bf16, Chips, m![R, Width]>,
        &mut HbmTensor<bf16, Chips, m![R, Width]>,
    ),
) -> Live<(), R> {
    type Tile<W, H, C> = tiles::Tile<bf16, W, H, C, tiles::Rows<W, H>>;
    type Landing<W, H, C> = (
        (Tile<W, H, C>, Tile<W, H, C>),
        (LandedTable<W, H, C>, LandedTable<W, H, C>),
    );
    let interval = input.with(());
    let (first_tile, count) = interval.groups::<tiles::Strip<Width, Head>>();
    if count == 0 {
        return interval;
    }
    let (query, key) = *input.value();
    let table = |device: &mut Device,
                 coefficients: &CoefficientRows<R, Head, Chips>,
                 at: usize,
                 landed: &mut LandedTable<Width, Head, Chips>| {
        // SAFETY: the checked interval stays in R; each tile is one complete row group.
        unsafe {
            coefficients.view().reshape::<Chips, m![{ tiles::S<R, Width, Head> }, { tiles::T<Width, Head> }, OutputHalf, { Symbol<Row<Head>> }]>()
        }
        .tile::<m![{ tiles::S<R, Width, Head> }], 1, m![{ tiles::S<R, Width, Head> } = 1 # ({ tiles::S<R, Width, Head> }), { tiles::T<Width, Head> }, OutputHalf, { Symbol<Row<Head>> }]>(at)
        .to_dm_view(&mut device.tdma, landed.view_mut());
    };
    let land = |device: &mut Device, at: usize, landing: &mut Landing<Width, Head, Chips>| {
        tiles::Layout::<R, Width, Head, Chips>::to_dm(query, device, at, &mut landing.0.0);
        tiles::Layout::<R, Width, Head, Chips>::to_dm(key, device, at, &mut landing.0.1);
        table(device, &rotation.low, at, &mut landing.1.0);
        table(device, &rotation.high, at, &mut landing.1.1);
    };
    let operand = |device: &mut Device,
                   landed: &LandedTable<Width, Head, Chips>|
     -> TileCoefficients<Width, Head, Chips> {
        device.sub.begin(landed.view())
            .fetch::<m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> } / 16], m![{ Symbol<Row<Head>> } % 16]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Lines<Width, Head> }, OutputHalf, { Symbol<Row<Head>> } / 8], m![{ Symbol<Row<Head>> } % 8]>()
            .to_vrf()
    };
    let rotate =
        |device: &mut Device,
         landing: &Landing<Width, Head, Chips>,
         outputs: &mut (Tile<Width, Head, Chips>, Tile<Width, Head, Chips>)| {
            let coefficients = (operand(device, &landing.1.0), operand(device, &landing.1.1));
            RotateTile::<Head>(core::marker::PhantomData)
                .forward(device, (&landing.0.0, &coefficients, &mut outputs.0));
            RotateTile::<Head>(core::marker::PhantomData)
                .forward(device, (&landing.0.1, &coefficients, &mut outputs.1));
        };
    let (rotated_query, rotated_key) = output;
    let mut store = |device: &mut Device,
                     rotated: &(Tile<Width, Head, Chips>, Tile<Width, Head, Chips>),
                     at: usize| {
        tiles::Layout::<R, Width, Head, Chips>::store(rotated_query, device, &rotated.0, at);
        tiles::Layout::<R, Width, Head, Chips>::store(rotated_key, device, &rotated.1, at);
    };
    let mut first: Landing<Width, Head, Chips> = (
        (DmTensor::new(), DmTensor::new()),
        (DmTensor::new(), DmTensor::new()),
    );
    let mut second: Landing<Width, Head, Chips> = (
        (DmTensor::new(), DmTensor::new()),
        (DmTensor::new(), DmTensor::new()),
    );
    let mut kept: (Tile<Width, Head, Chips>, Tile<Width, Head, Chips>) =
        (DmTensor::new(), DmTensor::new());
    let mut held: (Tile<Width, Head, Chips>, Tile<Width, Head, Chips>) =
        (DmTensor::new(), DmTensor::new());
    land(device, first_tile, &mut first);
    if count > 1 {
        land(device, first_tile + 1, &mut second);
    }
    for pair in 0..count / 2 {
        let at = 2 * pair;
        rotate(device, &first, &mut kept);
        land(device, first_tile + (at + 2).min(count - 1), &mut first);
        store(device, &kept, first_tile + at);
        rotate(device, &second, &mut held);
        land(device, first_tile + (at + 3).min(count - 1), &mut second);
        store(device, &held, first_tile + at + 1);
    }
    if count % 2 == 1 {
        rotate(device, &first, &mut kept);
        store(device, &kept, first_tile + count - 1);
    }
    interval
}
