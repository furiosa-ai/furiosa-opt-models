use furiosa_opt_std::prelude::*;

use crate::Module;
use crate::compose::Tiles;
use crate::models::nemo::serving::{Chips, Hidden, Rows};
use crate::ops::layout::Dm;
use crate::ops::linear::Affine;

mod contraction;
pub mod greedy;
pub mod sample;
pub type Vocabulary =
    crate::ops::vocabulary::readout::Local<crate::models::nemo::parameters::Readout>;
type Cluster = m![Vocabulary / 32768];
/// Each of a cluster's 256 slices owns 128 vocabulary rows with all of a tile's features, so no
/// pair of slices is reduced.
type Slices = m![Vocabulary % 32768 / 128];
type Source = m![Vocabulary % 32768 / 16384, Hidden % 256 / 16, Rows / 8];
/// The selected rows on every vocabulary slice group, each tile's 16-feature packets ready for the
/// broadcast into the contraction's register.
pub type Input = DmTensor<bf16, Chips, Cluster, Source, m![Hidden / 256, Rows % 8, Hidden % 16]>;
pub type Panel = DmTensor<bf16, Chips, Cluster, Slices, m![Vocabulary % 128, Hidden % 256]>;
type Register = TrfTensor<
    bf16,
    Chips,
    Cluster,
    Slices,
    m![Rows % 8],
    m![Hidden % 256 / 16, Rows / 8, Hidden % 16],
>;
pub type Values = DmTensor<f32, Chips, Cluster, Slices, m![Rows, Vocabulary % 128]>;
type Feed = m![
    Vocabulary % 128 / 4,
    Rows / 16,
    Vocabulary % 4 / 2,
    Hidden % 256 / 16
];
/// A weight packet: two adjacent vocabulary rows by 16 features, read as two 32-byte runs, so a
/// step contracts both rows against the register.
type Packet = m![Vocabulary % 2, Hidden % 16];
type Steps = m![{ Feed }, Rows % 16 / 8];
type Sums = m![
    Vocabulary % 128 / 4,
    Rows / 16,
    Vocabulary % 4 / 2,
    Rows % 16 / 8
];

/// The first 32 rows of a group: the contraction reads half the register.
type HalfFeed = m![
    Vocabulary % 128 / 4,
    Rows % 32 / 16,
    Vocabulary % 4 / 2,
    Hidden % 256 / 16
];
type HalfSteps = m![{ HalfFeed }, Rows % 16 / 8];
type HalfSums = m![
    Vocabulary % 128 / 4,
    Rows % 32 / 16,
    Vocabulary % 4 / 2,
    Rows % 16 / 8
];
pub type HalfValues = DmTensor<f32, Chips, Cluster, Slices, m![Rows % 32, Vocabulary % 128]>;

/// Hidden-tile-major, so each 256-feature tile lands as one contiguous read.
pub type Weights = HbmTensor<bf16, Chips, m![Hidden / 256, Vocabulary, Hidden % 256]>;
/// One 64-row group's logits of this chip's vocabulary shard.
pub type Logits = DmTensor<bf16, Chips, Cluster, Slices, m![Rows, Vocabulary % 128]>;

#[derive(DeviceSend)]
pub struct Projection<'a> {
    pub weight: HbmTensorView<'a, bf16, Chips, m![Hidden / 256, Vocabulary, Hidden % 256]>,
}

impl Projection<'_> {
    /// Hidden tile `tile` of the head weight, landed in data memory.
    fn land(&self, device: &mut Device, tile: usize) -> Panel {
        self.weight.tile::<m![Hidden / 256], 1,
            m![Hidden / 256 = 1 # (Hidden / 256), Vocabulary, Hidden % 256]>(tile)
            .to_dm(&mut device.tdma)
    }

    /// One group's logits, contracted as `C`.
    pub fn apply<C: Part>(&self, device: &mut Device, input: &Input) -> Logits {
        // Each tile's weight lands while the previous tile contracts, so the stream never waits
        // for the contraction and fold.
        let first = self.land(device, 0);
        let mut landed = self.land(device, 1);
        let mut total = C::project(device, input, (&first, 0));
        drop(first);
        #[unroll]
        for tile in 0..18 {
            let weight = landed;
            landed = self.land(device, tile + 2);
            let next = C::project(device, input, (&weight, tile + 1));
            total = C::add(device, (&total, &next));
        }
        let next = C::project(device, input, (&landed, 19));
        let total = C::add(device, (&total, &next));
        C::rounded(device, &total)
    }

    /// `apply` of two 64-row groups over one stream of the weight: each tile lands once and both
    /// groups contract it, so the stream that bounds one group is paid once per pair.
    pub fn apply_pair(&self, device: &mut Device, (a, b): (&Input, &Input)) -> (Logits, Logits) {
        let first = self.land(device, 0);
        let mut landed = self.land(device, 1);
        let mut total_a = Whole::project(device, a, (&first, 0));
        let mut total_b = Whole::project(device, b, (&first, 0));
        drop(first);
        #[unroll]
        for tile in 0..18 {
            let weight = landed;
            landed = self.land(device, tile + 2);
            let next = Whole::project(device, a, (&weight, tile + 1));
            total_a = add(device, (&total_a, &next));
            let next = Whole::project(device, b, (&weight, tile + 1));
            total_b = add(device, (&total_b, &next));
        }
        let next = Whole::project(device, a, (&landed, 19));
        let total_a = add(device, (&total_a, &next));
        let next = Whole::project(device, b, (&landed, 19));
        let total_b = add(device, (&total_b, &next));
        (rounded(device, &total_a), rounded(device, &total_b))
    }

    /// `apply_pair` of three groups, the last one contracted as `C`.
    pub fn apply_three<C: Part>(
        &self,
        device: &mut Device,
        (a, b, c): (&Input, &Input, &Input),
    ) -> (Logits, Logits, Logits) {
        let first = self.land(device, 0);
        let mut landed = self.land(device, 1);
        let mut total_a = Whole::project(device, a, (&first, 0));
        let mut total_b = Whole::project(device, b, (&first, 0));
        let mut total_c = C::project(device, c, (&first, 0));
        drop(first);
        #[unroll]
        for tile in 0..18 {
            let weight = landed;
            landed = self.land(device, tile + 2);
            let next = Whole::project(device, a, (&weight, tile + 1));
            total_a = add(device, (&total_a, &next));
            let next = Whole::project(device, b, (&weight, tile + 1));
            total_b = add(device, (&total_b, &next));
            let next = C::project(device, c, (&weight, tile + 1));
            total_c = C::add(device, (&total_c, &next));
        }
        let next = Whole::project(device, a, (&landed, 19));
        let total_a = add(device, (&total_a, &next));
        let next = Whole::project(device, b, (&landed, 19));
        let total_b = add(device, (&total_b, &next));
        let next = C::project(device, c, (&landed, 19));
        let total_c = C::add(device, (&total_c, &next));
        (
            rounded(device, &total_a),
            rounded(device, &total_b),
            C::rounded(device, &total_c),
        )
    }

    /// `apply_pair` of four groups, the last one contracted as `D`.
    pub fn apply_four<D: Part>(
        &self,
        device: &mut Device,
        (a, b, c, d): (&Input, &Input, &Input, &Input),
    ) -> (Logits, Logits, Logits, Logits) {
        let first = self.land(device, 0);
        let mut landed = self.land(device, 1);
        let mut total_a = Whole::project(device, a, (&first, 0));
        let mut total_b = Whole::project(device, b, (&first, 0));
        let mut total_c = Whole::project(device, c, (&first, 0));
        let mut total_d = D::project(device, d, (&first, 0));
        drop(first);
        #[unroll]
        for tile in 0..18 {
            let weight = landed;
            landed = self.land(device, tile + 2);
            let next = Whole::project(device, a, (&weight, tile + 1));
            total_a = add(device, (&total_a, &next));
            let next = Whole::project(device, b, (&weight, tile + 1));
            total_b = add(device, (&total_b, &next));
            let next = Whole::project(device, c, (&weight, tile + 1));
            total_c = add(device, (&total_c, &next));
            let next = D::project(device, d, (&weight, tile + 1));
            total_d = D::add(device, (&total_d, &next));
        }
        let next = Whole::project(device, a, (&landed, 19));
        let total_a = add(device, (&total_a, &next));
        let next = Whole::project(device, b, (&landed, 19));
        let total_b = add(device, (&total_b, &next));
        let next = Whole::project(device, c, (&landed, 19));
        let total_c = add(device, (&total_c, &next));
        let next = D::project(device, d, (&landed, 19));
        let total_d = D::add(device, (&total_d, &next));
        (
            rounded(device, &total_a),
            rounded(device, &total_b),
            rounded(device, &total_c),
            D::rounded(device, &total_d),
        )
    }
}

/// The rows of a group the head contracts: all of them, or the first 32 of a last group whose
/// other rows are not live.
pub trait Part {
    type Values;
    fn project(device: &mut Device, input: &Input, weight: (&Panel, usize)) -> Self::Values;
    fn add(device: &mut Device, terms: (&Self::Values, &Self::Values)) -> Self::Values;
    fn rounded(device: &mut Device, total: &Self::Values) -> Logits;
}

#[derive(Clone, Debug)]
pub struct Whole;
#[derive(Clone, Debug)]
pub struct Half;

impl Part for Whole {
    type Values = Values;

    fn project(device: &mut Device, input: &Input, (weight, tile): (&Panel, usize)) -> Values {
        let input = fill(device, input, tile);
        Affine::new(weight, ())
            .forward((
                device,
                Dm {
                    tiles: Tiles::<Whole>::over(Rows::SIZE),
                    value: &input,
                },
            ))
            .value
    }

    fn add(device: &mut Device, terms: (&Values, &Values)) -> Values {
        add(device, terms)
    }

    fn rounded(device: &mut Device, total: &Values) -> Logits {
        rounded(device, total)
    }
}

impl Part for Half {
    type Values = HalfValues;

    fn project(device: &mut Device, input: &Input, (weight, tile): (&Panel, usize)) -> HalfValues {
        let register = fill(device, input, tile);
        Affine::new(weight, ())
            .forward((
                device,
                Dm {
                    tiles: Tiles::<Half>::over(Rows::SIZE / 2),
                    value: &register,
                },
            ))
            .value
    }

    fn add(device: &mut Device, terms: (&HalfValues, &HalfValues)) -> HalfValues {
        add_half(device, terms)
    }

    /// Rows 32-63 of the group's logits stay unwritten.
    fn rounded(device: &mut Device, total: &HalfValues) -> Logits {
        rounded_half(device, total)
    }
}

/// Summed logits rounded to BF16.
fn rounded(device: &mut Device, total: &Values) -> Logits {
    device
        .main
        .begin(total.view())
        .fetch::<m![Rows, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .collect::<m![Rows, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .cast::<bf16, m![Vocabulary % 8 # 16]>()
        .commit_trim::<m![Vocabulary % 8]>()
        .commit()
}

/// Two tiles' partial logits summed.
fn add(device: &mut Device, (total, next): (&Values, &Values)) -> Values {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(total.view(), next.view())
        .fetch::<m![Rows / 2, Term, Rows % 2, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .collect::<m![Rows / 2, Term, Rows % 2, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows, Vocabulary % 128 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![Vocabulary % 8]>()
        .commit()
}

/// Tile `tile` of the group's input in the register file.
fn fill(device: &mut Device, input: &Input, tile: usize) -> Register {
    let input = input
        .view()
        .tile::<m![Hidden / 256], 1, m![Hidden / 256 = 1 # (Hidden / 256), Rows % 8, Hidden % 16]>(
            tile,
        );
    device
        .sub
        .begin(input)
        .fetch::<m![Rows % 8], m![Hidden % 16]>()
        .switch::<Slices, m![Rows % 8, Hidden % 256 / 16, Rows / 8]>(
            SwitchConfig::TransposedBroadcast1 {
                slice1: 1,
                slice0: 128,
            },
        )
        .collect::<m![Rows % 8, Hidden % 256 / 16, Rows / 8], m![Hidden % 16]>()
        .to_trf()
}

/// `rounded` of a group's first 32 rows, its other rows left unwritten.
fn rounded_half(device: &mut Device, total: &HalfValues) -> Logits {
    let mut logits = Logits::new();
    device
        .main
        .begin(total.view())
        .fetch::<m![Rows % 32, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .collect::<m![Rows % 32, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .cast::<bf16, m![Vocabulary % 8 # 16]>()
        .commit_trim::<m![Vocabulary % 8]>()
        .commit_view(logits.view_mut().tile::<m![Rows / 32], 1, m![
            Rows / 32 = 1 #{!} 2,
            Rows % 32,
            Vocabulary % 128
        ]>(0));
    logits
}

/// `add` of a group's first 32 rows.
fn add_half(device: &mut Device, (total, next): (&HalfValues, &HalfValues)) -> HalfValues {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(total.view(), next.view())
        .fetch::<m![Rows % 32 / 2, Term, Rows % 2, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .collect::<m![Rows % 32 / 2, Term, Rows % 2, Vocabulary % 128 / 8], m![Vocabulary % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Rows % 32, Vocabulary % 128 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![Vocabulary % 8]>()
        .commit()
}
