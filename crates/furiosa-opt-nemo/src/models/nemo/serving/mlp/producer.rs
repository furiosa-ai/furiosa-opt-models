use crate::Module;
use furiosa_opt_std::prelude::*;

pub(super) mod contraction;

use super::{Branch, Branches, Halves, Hidden, Intermediate, Rows, Waves, down, weight};
use crate::models::nemo::serving::Chips;

type Cluster = m![Intermediate / 3584];
type Source = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    m![Intermediate % 3584 / 896, Hidden % 2560 / 160, Rows / 16],
    m![Hidden / 2560, Rows % 16, Hidden % 160],
>;
type Slice = m![Intermediate % 3584 / 56, Hidden % 2560 / 640];
type Lanes = m![Rows % 8];
type Feed = m![Intermediate % 56 / 2, Hidden % 640 / 160, Hidden % 160 / 32];
type Steps = m![{ Feed }, Rows / 16];
type Flit = m![Intermediate % 2, Hidden % 32];
type Sums = m![Intermediate % 56 / 2, Rows / 16];
type Groups = m![{ Sums }, Intermediate % 2];
type PairFeed = m![Branch, { Feed }];
type PairSteps = m![{ PairFeed }, Rows / 16];
type PairSums = m![Branch, { Sums }];
type PairGroups = m![{ PairSums }, Intermediate % 2];
pub type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![
        Hidden % 160 / 32,
        Hidden % 640 / 160,
        Rows / 16,
        Hidden % 32
    ],
>;
/// One wave's activation: each 160-wide hidden group holds the wave's rows back to back, so the
/// norm stores four rows and a gate slice lands sixteen in one run.
pub type Grouped = m![Hidden / 160, Rows, Hidden % 160];
/// The gate/up activation of every wave, quantized by the gate's input scale.
pub type Input = HbmTensor<f8e4m3, Chips, m![Waves, { Grouped }]>;

/// The quantized activation of one wave, landed once for the gate and up halves it feeds.
pub struct Activation(Source);

impl Activation {
    pub fn land(device: &mut Device, input: &Input, wave: usize) -> Self {
        let mut activation = Self::new();
        activation.land_into(device, input, wave);
        activation
    }

    /// An activation not yet landed, for `land_into`.
    pub fn new() -> Self {
        Self(Source::new())
    }

    pub fn land_into(&mut self, device: &mut Device, input: &Input, wave: usize) {
        input
            .view()
            .tile::<m![Waves], 1, m![Waves = 1 # Waves, { Grouped }]>(wave)
            .to_dm_view(&mut device.tdma, self.0.view_mut());
    }

    /// Branch `branch` of hidden half `half`'s projection into that branch of `output`, as wave 0
    /// contracts each weight half as it lands.
    pub fn project(
        &self,
        device: &mut Device,
        (weight, branch): (&weight::GateUp, usize),
        (half, output): (usize, &mut Branches),
    ) {
        #[unroll]
        for lane in 0..2 {
            let input = self.fill(device, (half, lane));
            contract_branch(device, (weight, branch, &input), (&mut *output, lane));
        }
    }

    /// The gate and up projections of both hidden halves, one command a 32-row register fill; a
    /// fill issued behind a command loads under it.
    pub fn project_halves(&self, device: &mut Device, pairs: &[weight::GateUp; 2]) -> Halves {
        let mut halves = (Branches::new(), Branches::new());
        let first = self.fill(device, (0, 0));
        contract(device, (&pairs[0], &first), (&mut halves.0, 0));
        let second = self.fill(device, (0, 1));
        contract(device, (&pairs[0], &second), (&mut halves.0, 1));
        let third = self.fill(device, (1, 0));
        contract(device, (&pairs[1], &third), (&mut halves.1, 0));
        let fourth = self.fill(device, (1, 1));
        contract(device, (&pairs[1], &fourth), (&mut halves.1, 1));
        halves
    }

    /// Hidden half `half` of the wave's rows `16k + 8 lane .. 16k + 8 lane + 8` in the register
    /// file, each slice holding its K group's 640 features.
    pub fn fill(&self, device: &mut Device, (half, lane): (usize, usize)) -> Register {
        device
            .sub
            .begin(
                self.0
                    .view()
                    .tile::<m![Hidden / 2560], 1, m![Hidden / 2560 = 1 # 2, Rows % 16, Hidden % 160]>(
                        half,
                    )
                    .tile::<m![Rows % 16 / 8], 1, m![
                        Hidden / 2560 = 1 # 2,
                        Rows % 16 / 8 = 1 # 2,
                        Rows % 8,
                        Hidden % 160
                    ]>(lane),
            )
            .fetch::<m![Rows % 8, Hidden % 160 / 32], m![Hidden % 32]>()
            .switch::<Slice, m![Rows % 8, Hidden % 160 / 32, Hidden % 640 / 160, Rows / 16]>(
                SwitchConfig::TransposedBroadcast1 {
                    slice0: 16,
                    slice1: 4,
                },
            )
            .collect::<m![Rows % 8, Hidden % 160 / 32, Hidden % 640 / 160, Rows / 16], m![Hidden % 32]>()
            .to_trf()
    }
}

/// The rows of a wave its gate and up contractions read: all of them, or the first 32 of a last
/// wave whose other rows are not live. A register's contraction is issued as `begin`, the next
/// register's fill, then `end`, so that fill loads under the register's first command.
pub trait Part {
    /// The previous wave's `raw` product relaid ahead of this wave's first contraction, where
    /// Main waits for the first fill, or `None` to relay it after the third.
    fn relay(device: &mut Device, raw: &down::Raw) -> Option<down::Activation>;

    fn begin(
        device: &mut Device,
        operands: (&weight::GateUp, &Register),
        output: (&mut Branches, usize),
    );

    fn end(
        device: &mut Device,
        operands: (&weight::GateUp, &Register),
        output: (&mut Branches, usize),
    );
}

/// All rows, gate and up in one command a register: the wave is Main-bound, and the relayout
/// runs under its first fill.
pub struct Whole;
/// The first 32 rows, gate and up a command each with the next fill between them: the wave is
/// bound by its fills, which a two-branch command over half the rows holds back.
pub struct Half;

impl Part for Whole {
    fn relay(device: &mut Device, raw: &down::Raw) -> Option<down::Activation> {
        Some(down::relayout(device, raw))
    }

    fn begin(
        device: &mut Device,
        operands: (&weight::GateUp, &Register),
        output: (&mut Branches, usize),
    ) {
        contract(device, operands, output);
    }

    fn end(_: &mut Device, _: (&weight::GateUp, &Register), _: (&mut Branches, usize)) {}
}

type HalfSteps = m![{ Feed }, Rows % 32 / 16];
type HalfSums = m![Intermediate % 56 / 2, Rows % 32 / 16];
type HalfGroups = m![{ HalfSums }, Intermediate % 2];
/// `Values`' slices with the row groups past the first 32 rows as padding.
type HalfSlice = m![Intermediate % 3584 / 56, Rows % 32 / 16 # 4];

impl Part for Half {
    fn relay(_: &mut Device, _: &down::Raw) -> Option<down::Activation> {
        None
    }

    fn begin(
        device: &mut Device,
        (weight, input): (&weight::GateUp, &Register),
        output: (&mut Branches, usize),
    ) {
        contract_half(device, (weight, 0, input), output);
    }

    fn end(
        device: &mut Device,
        (weight, input): (&weight::GateUp, &Register),
        output: (&mut Branches, usize),
    ) {
        contract_half(device, (weight, 1, input), output);
    }
}

/// One lane's contraction of branch `branch` of `weight` over the register's first 32 rows,
/// committed into that branch of `output`'s lane.
fn contract_half(
    device: &mut Device,
    (weight, branch, input): (&weight::GateUp, usize, &Register),
    (output, lane): (&mut Branches, usize),
) {
    // SAFETY: row group `Rows / 16` below 2 is `Rows % 32 / 16`, so naming the slices by it
    // moves no data; the slices of row groups 2 and 3 keep what they held.
    let output = unsafe {
        output.view_mut().reshape::<Chips, Cluster, HalfSlice, m![
            Branch,
            Intermediate % 56,
            Rows % 16 / 8,
            Rows % 8
        ]>()
    };
    let projection = crate::ops::linear::Affine::new(
        weight
            .view()
            .tile::<m![Branch], 1, m![Branch = 1 # 2, Intermediate % 56, Hidden % 640]>(branch),
        (),
    );
    projection
        .forward((
            device,
            crate::ops::layout::Dm::<contraction::Partial, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (input,),
            },
        ))
        .value
        .commit_trim::<Lanes>()
        .commit_view(
            output
                .tile::<m![Branch], 1, m![
                    Branch = 1 #{!} 2,
                    Intermediate % 56,
                    Rows % 16 / 8,
                    Rows % 8
                ]>(branch)
                .tile::<m![Rows % 16 / 8], 1, m![
                    Branch = 1 #{!} 2,
                    Intermediate % 56,
                    Rows % 16 / 8 = 1 #{!} 2,
                    Rows % 8
                ]>(lane),
        );
}

/// One lane's gate and up contractions of `weight` against the filled register in one command,
/// committed into `output`'s lane.
pub(super) fn contract(
    device: &mut Device,
    (weight, input): (&weight::GateUp, &Register),
    (output, lane): (&mut Branches, usize),
) {
    let projection = crate::ops::linear::Affine::new(weight.view(), ());
    projection
        .forward((
            device,
            crate::ops::layout::Dm::<contraction::Paired, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (input,),
            },
        ))
        .value
        .commit_trim::<Lanes>()
        .commit_view(output.view_mut().tile::<m![Rows % 16 / 8], 1, m![
            Branch,
            Intermediate % 56,
            Rows % 16 / 8 = 1 #{!} 2,
            Rows % 8
        ]>(lane));
}

/// One lane's contraction of branch `branch` of `weight` against the filled register, committed
/// into that branch of `output`'s lane.
fn contract_branch(
    device: &mut Device,
    (weight, branch, input): (&weight::GateUp, usize, &Register),
    (output, lane): (&mut Branches, usize),
) {
    let projection = crate::ops::linear::Affine::new(
        weight
            .view()
            .tile::<m![Branch], 1, m![Branch = 1 # 2, Intermediate % 56, Hidden % 640]>(branch),
        (),
    );
    projection
        .forward((
            device,
            crate::ops::layout::Dm::<contraction::Single, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (input,),
            },
        ))
        .value
        .commit_trim::<Lanes>()
        .commit_view(
            output
                .view_mut()
                .tile::<m![Branch], 1, m![
                    Branch = 1 #{!} 2,
                    Intermediate % 56,
                    Rows % 16 / 8,
                    Rows % 8
                ]>(branch)
                .tile::<m![Rows % 16 / 8], 1, m![
                    Branch = 1 #{!} 2,
                    Intermediate % 56,
                    Rows % 16 / 8 = 1 #{!} 2,
                    Rows % 8
                ]>(lane),
        );
}
