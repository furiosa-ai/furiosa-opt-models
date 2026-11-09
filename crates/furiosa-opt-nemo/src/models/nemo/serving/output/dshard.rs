//! The O projection with the layer D-sharded: chip `c` projects its own hidden half
//! (`Hidden / 2560 = c`) from every head's FP8 attended rows, so its output is final and the
//! only O transfer is the peer's 2,048 head features (FP8) of each row. Cluster `k` owns the
//! 640-wide panels of parity `k` (`Hidden % 1280 / 640`, as the D-sharded feed-forward's
//! panels), both 1,280-wide halves of its chip half (`Hidden % 2560 / 1280`) in 80-wide groups.
//! The own heads and the peer's heads are two contractions (`Part`), each rounded to BF16 and
//! then summed and rounded again, as TP2 rounds each chip's partial and their sum, so the output
//! is TP2's O sum of the own half to the bit. The weight is the TP2 weight's bytes regrouped per
//! chip on the host, 10 KiB a slice a part in TRF.
use crate::Module;
use furiosa_opt_std::prelude::*;

use crate::models::nemo::serving::{Chips, Hidden, Rows, Width, scale};

axes![Part = 2, Term = 2];
/// Cluster `k` owns the 640-wide panels of parity `k` of every 1,280 features.
pub type Cluster = m![Hidden % 1280 / 640];
type Slice = m![Hidden % 2560 / 1280, Hidden % 640 / 80, Width / 128];
/// A slice of one part's rows: four rows of one 128-feature group.
pub type Landing = m![Rows / 4, Width / 128];
pub type Reduced = m![Hidden % 2560 / 1280, Hidden % 640 / 80, Rows / 4];
pub type Lanes = m![Hidden % 8];
type Tile = m![Width % 128 / 32, Hidden % 80 / 8, Width % 32];
type FillTime = m![Hidden % 8, Width % 128 / 32, Hidden % 80 / 8];
type FillPacket = m![Width % 32];
type GatherTime = m![Rows / 4];
type InputPacket = m![Width % 128 / 32, Rows % 4, Width % 32];
type Time = m![Rows / 4, Width % 128 / 32, Rows % 4 / 2];
type Steps = m![{ Time }, Hidden % 80 / 8];
type Packet = m![Rows % 2, Width % 32];
type Flits = m![{ Time }, Rows % 2];
type Flit = m![Width % 32];
type Sums = m![{ GatherTime }, Rows % 4 / 2, Hidden % 80 / 8];
type Groups = m![{ Sums }, Rows % 2];
type OutputTime = m![Rows % 4 / 2, Hidden % 80 / 8, Rows % 2];
pub type PartialTime = m![Rows % 4, Hidden % 80 / 8];
/// A slice's four rows of one 128-feature group, one 128-byte run each.
pub type Element = m![Rows % 4, Width % 128];
/// Every layer's weight: a chip's own hidden half by its own heads (part 0) and by the peer's
/// heads (part 1).
pub type Stored = m![Part, { Cluster }, { Slice }, { Lanes }, { Tile }];
/// One wave's FP8 rows of one part's heads, on both clusters.
pub type Rows8 = DmTensor<f8e4m3, Chips, Cluster, Landing, Element>;
/// One wave's attended rows of this chip's heads before quantization.
pub type Rows32 = DmTensor<f32, Chips, Cluster, Landing, Element>;
/// One wave's own-half output a cluster: four rows of an 80-wide group a slice.
pub type Output = DmTensor<bf16, Chips, Cluster, Reduced, m![{ PartialTime }, { Lanes }]>;
axes![Wave = 16];
type Space = crate::compose::Tile<Rows, Wave>;

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = Element;
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
        slice1: 16,
        slice0: 16,
    };
}

const _: () = {
    assert!(Slice::SIZE == 256);
    assert!(Landing::SIZE == 256);
    assert!(Reduced::SIZE == 256);
    assert!(Tile::SIZE * Lanes::SIZE == 10240);
    assert!(Stored::SIZE == 2 * Width::SIZE * Hidden::SIZE / 2);
};

type Register = TrfTensor<f8e4m3, Chips, Cluster, Slice, Lanes, Tile>;

fn fill_part(
    device: &mut Device,
    weight: &DmTensor<f8e4m3, Chips, Cluster, Slice, m![{ Lanes }, { Tile }]>,
) -> Register {
    device
        .sub
        .begin(weight.view())
        .fetch::<FillTime, FillPacket>()
        .collect::<FillTime, FillPacket>()
        .to_trf()
}

/// One layer's own-half weight (both parts) and its scales in data memory.
pub struct Weight {
    weight: [DmTensor<f8e4m3, Chips, Cluster, Slice, m![{ Lanes }, { Tile }]>; 2],
    scales: scale::Scales<Chips>,
}

/// Both parts in the register file with the scales their epilogue reads.
pub struct Filled {
    registers: [Register; 2],
    epilogue: scale::Epilogue<Cluster, Reduced, Chips>,
}

impl Weight {
    pub fn new() -> Self {
        Self {
            weight: [DmTensor::new(), DmTensor::new()],
            scales: scale::Scales::new(),
        }
    }

    /// The weight (two 10-KiB runs a slice) and its packed scale row.
    pub fn land_from<F: M>(
        &mut self,
        device: &mut Device,
        (weight, scales): (
            &HbmTensor<f8e4m3, Chips, Stored>,
            HbmTensorView<'_, f32, Chips, F>,
        ),
    ) {
        #[unroll]
        for part in 0..2 {
            weight
                .view()
                .tile::<m![Part], 1, m![Part = 1 # 2, { Cluster }, { Slice }, { Lanes }, { Tile }]>(
                    part,
                )
                .to_dm_view(&mut device.tdma, self.weight[part].view_mut());
        }
        self.scales = scale::Scales::land(device, scales);
    }

    pub fn fill(&self, device: &mut Device) -> Filled {
        let own = fill_part(device, &self.weight[0]);
        let peer = fill_part(device, &self.weight[1]);
        Filled {
            registers: [own, peer],
            epilogue: scale::Epilogue {
                input: self.scales.get(device, 0),
                weight: self.scales.get(device, 1),
            },
        }
    }

    /// The O input scale in every slice of a 256-slice layout.
    pub fn divisor<C: M, S: M>(&self, device: &mut Device) -> scale::InputScale<Chips, C, S> {
        crate::ops::linear::Affine::new((), self.scales.get(device, 0))
    }
}

type QuantizeTime = m![Rows % 4, Width % 128 / 8];
type QuantizePacket = m![Width % 8];

/// One wave's own heads rounded once to BF16 and quantized by the O input scale `divisor`.
pub fn quantize(
    device: &mut Device,
    rows: &Rows32,
    divisor: &scale::InputScale<Chips, Cluster, Landing>,
) -> Rows8 {
    let rounded: DmTensor<bf16, Chips, Cluster, Landing, m![{ QuantizeTime }, { QuantizePacket }]> =
        device
            .main
            .begin(rows.view())
            .fetch::<QuantizeTime, QuantizePacket>()
            .collect::<QuantizeTime, QuantizePacket>()
            .cast::<bf16, m![{ QuantizePacket } # 16]>()
            .commit_trim::<QuantizePacket>()
            .commit();
    let quantized = scale::quantize::<Cluster, Landing, QuantizeTime, QuantizePacket, Chips>(
        device, &rounded, divisor,
    );
    // SAFETY: `[Rows % 4, Width % 128 / 8, Width % 8]` is the element in wire order.
    unsafe { quantized.reshape() }
}

impl Filled {
    /// One part's contraction of one wave's rows, scaled, rounded to BF16.
    fn project(&self, device: &mut Device, (rows, part): (&Rows8, usize)) -> Output {
        crate::ops::linear::Affine::new(&self.registers[part], &self.epilogue)
            .forward((
                device,
                crate::ops::layout::Dm::<Space, _> {
                    tiles: crate::compose::Tiles::over(Rows::SIZE),
                    value: rows,
                },
            ))
            .value
    }

    /// One wave's own-half output from its own heads' rows and the peer's, into `output`.
    pub fn project_into<E: M>(
        &self,
        device: &mut Device,
        (own, peer): (&Rows8, &Rows8),
        output: DmTensorViewMut<'_, bf16, Chips, Cluster, Reduced, E>,
    ) {
        let first = self.project(device, (own, 0));
        let second = self.project(device, (peer, 1));
        type Blocks = m![{ PartialTime } / 20, Term, { PartialTime } % 20];
        const { assert!(PartialTime::SIZE % 20 == 0) };
        device
            .main
            .begin_interleaved::<Term, _, _, _, _, _>(first.view(), second.view())
            .fetch::<Blocks, Lanes>()
            .fetch_cast::<f32>()
            .collect::<Blocks, Lanes>()
            .vector_init()
            .vector_intra_slice_unzip::<Term, PartialTime>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![{ Lanes } # 16]>()
            .commit_trim::<Lanes>()
            .commit_view(output)
    }
}
