use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::stack::Depth;
use crate::models::nemo::serving::{Chips, Hidden, Rows, Width, scale, stack};

pub mod dshard;
pub mod oblock;

pub type Cluster = m![Hidden / 2560];
type Slice = m![Hidden % 2560 / 80, Width / 256];
pub type Landing = m![Rows / 2, Width / 256];
pub type Reduced = m![Hidden % 2560 / 80, Rows / 8];
pub type Lanes = m![Hidden % 8];
type Tile = m![Width % 256 / 32, Hidden % 80 / 8, Width % 32];
/// A slice's 80 outputs by 256 heads' features as landed: four 5,120-byte runs of `Stored`.
type WeightElement = m![
    Width % 256 / 64,
    Hidden % 80 / 8,
    Width % 64 / 8,
    Hidden % 8,
    Width % 8
];
type FillTime = m![Hidden % 8, Width % 256 / 32, Hidden % 80 / 8];
type FillPacket = m![Width % 32];
type GatherTime = m![Rows / 2];
type InputPacket = m![Width % 256 / 32, Rows % 2, Width % 32];
type Time = m![Rows / 2, Width % 256 / 32];
type Steps = m![{ Time }, Hidden % 80 / 8];
type Packet = m![Rows % 2, Width % 32];
type Flits = m![{ Time }, Rows % 2];
type Flit = m![Width % 32];
type Sums = m![{ GatherTime }, Hidden % 80 / 8];
type Groups = m![{ Sums }, Rows % 2];
pub type InputTime = m![Rows % 2, Width % 256 / 8];
type OutputTime = m![Rows % 8 / 2, Hidden % 80 / 8, Rows % 2];
pub type PartialTime = m![Rows % 8, Hidden % 80 / 8];
axes![Slot = 3];
/// Every layer's O weight on a chip, the one copy both routes read: slot `h - c + 1` holds hidden
/// half `h` by chip `c`'s heads, for the three pairs the chip reads. Decode reads slots
/// `1 - rank ..` (its heads by both halves), the D-sharded prompt route slots `rank ..` (its half
/// by chip 1's heads, then chip 0's: `oblock::Stored`).
pub type Stored = m![Slot, { Half }];
/// One hidden half's 2,560 outputs by one chip's 2,048 head features.
type Half = m![
    Hidden % 1280 / 640,
    Hidden % 640 / 320,
    Hidden % 320 / 160,
    Width / 64,
    Hidden % 2560 / 1280,
    Hidden % 160 / 80,
    Hidden % 80 / 8,
    Width % 64 / 8,
    Hidden % 8,
    Width % 8
];
pub type Input = HbmTensor<bf16, Chips, m![Rows, Width]>;
pub type Partial = DmTensor<bf16, Chips, Cluster, Reduced, m![{ PartialTime }, { Lanes }]>;
axes![Wave = 8];
type Space = crate::compose::Tile<Rows, Wave>;

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = m![{ InputTime }, Width % 8];
    type Lanes = Lanes;
    type Weight = Tile;
    type Feed = m![1];
    type Packet = Packet;
    type Flits = Flits;
    type Flit = Flit;
    type Steps = Steps;
    type Contracted = m![Rows % 2];
    type Sums = Sums;
    type Groups = Groups;
    type Output = OutputTime;
}

impl crate::ops::linear::contraction::broadcast::Rows for Space {
    type Result = m![{ PartialTime }, { Lanes }];
    type Landing = Landing;
    type InputPacket = InputPacket;
    type Gather = GatherTime;
    const BROADCAST: SwitchConfig = SwitchConfig::Broadcast1 {
        slice1: 32,
        slice0: 8,
    };
}

const _: () = {
    assert!(Slice::SIZE == 256);
    assert!(Landing::SIZE == 256);
    assert!(Reduced::SIZE == 256);
    assert!(2 * Stored::SIZE == 3 * Hidden::SIZE * Width::SIZE);
};

/// The `Stored` position of O value (absolute `hidden` output, absolute head `feature`) on every
/// chip that holds it: its hidden half's chip and its heads' chip, the same slot on both.
pub fn places(hidden: usize, feature: usize) -> ([usize; 2], usize) {
    let (half, owner) = (hidden / 2560, feature / 2048);
    let (h, w) = (hidden % 2560, feature % 2048);
    let index = [
        (3, half + 1 - owner),
        (2, h % 1280 / 640),
        (2, h % 640 / 320),
        (2, h % 320 / 160),
        (32, w / 64),
        (2, h / 1280),
        (2, h % 160 / 80),
        (10, h % 80 / 8),
        (8, w % 64 / 8),
        (8, h % 8),
        (8, w % 8),
    ]
    .iter()
    .fold(0, |index, &(size, digit)| index * size + digit);
    ([half, owner], index)
}

/// One layer's O weight and scales in data memory.
pub struct Weight {
    weight: DmTensor<f8e4m3, Chips, Cluster, Slice, WeightElement>,
    scales: scale::Scales<Chips>,
}

impl Weight {
    /// A weight not yet landed, for `land` then `spread` to fill where the layer first needs it.
    pub fn new() -> Self {
        Self {
            weight: DmTensor::new(),
            scales: scale::Scales::new(),
        }
    }

    /// The weight's and its scales' transfers into `seeds`, without the Main work that spreads
    /// the scales: chip `rank`'s heads by both hidden halves, slots `1 - rank ..` of `Stored`.
    pub fn land(
        &mut self,
        device: &mut Device,
        (layers, index, rank): (&stack::Layers, usize, usize),
        seeds: &mut scale::Seeds<Chips>,
    ) {
        let window = layers
            .output
            .at(index)
            .tile::<m![Slot], 2, m![Depth = 1 # Depth, Slot = 2 # 3, { Half }]>(1 - rank);
        // SAFETY: slot `1 - rank + h` holds hidden half `h` by this chip's heads, so naming the
        // window's slot axis by the half moves no data.
        let halves = unsafe {
            window.reshape::<Chips, m![Depth = 1 # Depth, Hidden / 2560 # 3, { Half }]>()
        };
        halves.to_dm_view(&mut device.tdma, self.weight.view_mut());
        seeds.land_into(device, layers.output_scales.at(index));
    }

    pub fn spread(&mut self, device: &mut Device, seeds: &scale::Seeds<Chips>) {
        self.scales.spread(device, seeds);
    }
}

/// A slice's attended rows: two rows of its 256-wide group of heads.
pub type Element = m![{ InputTime }, Width % 8];
/// One 64-row wave's attended rows in BF16, replicated on both clusters.
pub type Landed = DmTensor<bf16, Chips, Cluster, Landing, Element>;

/// One layer's O weight in the register file; nothing of it depends on the wave, so it fills while
/// the wave's rows are still landing.
pub struct Filled(TrfTensor<f8e4m3, Chips, Cluster, Slice, Lanes, Tile>);

/// The O projection's scales in registers: its rows' input scale and the epilogue's pair.
pub struct Factors {
    divisor: scale::InputScale<Chips, Cluster, Landing>,
    epilogue: scale::Epilogue<Cluster, Reduced, Chips>,
}

impl Weight {
    pub fn fill(&self, device: &mut Device) -> Filled {
        Filled(
            device
                .sub
                .begin(self.weight.view())
                .fetch::<FillTime, FillPacket>()
                .collect::<FillTime, FillPacket>()
                .to_trf(),
        )
    }

    /// The spread scales in registers, for every projection of the layer that reads them.
    pub fn factors(&self, device: &mut Device) -> Factors {
        Factors {
            divisor: crate::ops::linear::Affine::new((), self.scales.get(device, 0)),
            epilogue: scale::Epilogue {
                input: self.scales.get(device, 0),
                weight: self.scales.get(device, 1),
            },
        }
    }
}

impl Filled {
    /// `project` into `output`, a view of a wider buffer.
    pub fn project_into<E: M>(
        &self,
        device: &mut Device,
        (rows, factors): (&Landed, &Factors),
        output: DmTensorViewMut<'_, bf16, Chips, Cluster, Reduced, E>,
    ) {
        let rows = scale::quantize::<Cluster, Landing, InputTime, m![Width % 8], Chips>(
            device,
            rows,
            &factors.divisor,
        );
        crate::ops::linear::Affine::new(&self.0, &factors.epilogue).forward((
            device,
            crate::ops::layout::Dm::<Space, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (&rows, output),
            },
        ));
    }

    /// One 64-row wave's O-projection partial over this chip's heads.
    pub fn project(&self, device: &mut Device, (rows, factors): (&Landed, &Factors)) -> Partial {
        let rows = scale::quantize::<Cluster, Landing, InputTime, m![Width % 8], Chips>(
            device,
            rows,
            &factors.divisor,
        );
        crate::ops::linear::Affine::new(&self.0, &factors.epilogue)
            .forward((
                device,
                crate::ops::layout::Dm::<Space, _> {
                    tiles: crate::compose::Tiles::over(Rows::SIZE),
                    value: &rows,
                },
            ))
            .value
    }
}
