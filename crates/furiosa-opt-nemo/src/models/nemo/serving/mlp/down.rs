use crate::Module;
use furiosa_opt_std::prelude::*;

use super::{Hidden, Intermediate, Product, Rows, Waves};
use crate::models::nemo::serving::step::{self, HiddenRows};
use crate::models::nemo::serving::{Chips, residual, scale, stack, stack::Depth};

type Cluster = m![Intermediate # 8192 / 4096];
pub type Half = m![Hidden / 2560];
type Slice = m![Hidden % 2560 / 80, Intermediate # 8192 % 2048 / 256];
pub type Reduced = m![Hidden % 2560 / 80, Rows / 8];
type Lanes = m![Rows % 8];
type Feed = m![
    Hidden % 80 / 4,
    Hidden / 2560,
    Hidden % 4 / 2,
    Rows / 16,
    Intermediate # 8192 % 256 / 64,
    Intermediate # 8192 % 64 / 32
];
type Steps = m![{ Feed }, Rows % 16 / 8];
type Packet = m![Hidden % 2, Intermediate # 8192 % 32];
type Sums = m![Hidden % 80 / 4, Hidden / 2560, Hidden % 4 / 2, Rows / 8];
type Groups = m![{ Sums }, Hidden % 2];
axes![Wave = 8];
type Space = crate::compose::Tile<Rows, Wave>;

impl crate::ops::linear::contraction::Mapping for Space {
    type Cluster = Cluster;
    type Slices = Slice;
    type Reduced = Reduced;
    type Input = m![
        Intermediate # 8192 % 64 / 32,
        Intermediate # 8192 % 256 / 64,
        Rows / 8,
        Intermediate # 8192 % 32
    ];
    type Lanes = Lanes;
    type Weight = m![Hidden / 2560, Hidden % 80, Intermediate # 8192 % 256];
    type Feed = Feed;
    type Packet = Packet;
    type Flits = m![{ Feed }, Hidden % 2];
    type Flit = m![Intermediate # 8192 % 32];
    type Steps = Steps;
    type Contracted = m![Hidden % 2];
    type Sums = Sums;
    type Groups = Groups;
    type Output = m![Hidden % 80 / 4, Hidden / 2560, Hidden % 4];
}
/// The down weight in the D-sharded prompt route's layout, the only stored form: 80 outputs by
/// 256 features one 20-KiB run, a decode slice's two hidden halves of one K panel one stride apart.
pub type Stored = m![
    Hidden / 1280,
    Hidden % 1280 / 640,
    Hidden % 640 / 320,
    Intermediate # 8192 / 4096,
    Intermediate # 8192 % 4096 / 2048,
    Intermediate # 8192 % 2048 / 256,
    Hidden % 320 / 80,
    Hidden % 80,
    Intermediate # 8192 % 256
];
pub(super) type Weight = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    m![Hidden / 2560, Hidden % 80, Intermediate # 8192 % 256],
>;
pub type Activation = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    m![Intermediate # 8192 % 2048 / 64, Rows / 32, Rows % 32 / 16, Rows % 16 / 8],
    m![Intermediate # 8192 % 4096 / 2048, Rows % 8, Intermediate # 8192 % 64],
>;
pub type Register = TrfTensor<
    f8e4m3,
    Chips,
    Cluster,
    Slice,
    Lanes,
    m![
        Intermediate # 8192 % 64 / 32,
        Intermediate # 8192 % 256 / 64,
        Rows / 8,
        Intermediate # 8192 % 32
    ],
>;
pub type Values = DmTensor<f32, Chips, Cluster, Reduced, m![Hidden / 2560, Rows % 8, Hidden % 80]>;
/// `Values` rounded to BF16, so half the bytes cross the clusters.
pub type Folded = DmTensor<bf16, Chips, Cluster, Reduced, m![Hidden / 2560, Rows % 8, Hidden % 80]>;
pub type Raw = DmTensor<
    f8e4m3,
    Chips,
    Cluster,
    m![
        Intermediate # 8192 % 2048 / 64,
        Rows / 32,
        Intermediate # 8192 % 4096 / 2048,
        Intermediate # 8192 % 64 / 32
    ],
    m![
        Intermediate # 8192 % 32 / 8,
        Rows % 32 / 16,
        Intermediate # 8192 % 8,
        Rows % 16 / 8,
        Rows % 8
    ],
>;
/// Each chip's BF16 partial: every cluster stores the hidden half it owns after the exchange,
/// in the step's slice-blocked hidden layout.
pub type Output = HbmTensor<bf16, Chips, m![Waves, { Half }, { Reduced }, Rows % 8, Hidden % 80]>;
pub type PartialTime = m![Rows % 8, Hidden % 80 / 8];
pub type PartialPacket = m![Hidden % 8];
/// One wave's BF16 partial of this chip in the step's residual layout.
pub type Partial = DmTensor<bf16, Chips, Half, Reduced, m![{ PartialTime }, { PartialPacket }]>;

/// The first down panel, landed while the gate/up tail still runs.
pub(super) struct Panel(Weight);

/// Both down panels in data memory.
pub struct Panels<'a> {
    pub(super) left: &'a Weight,
    pub(super) right: Weight,
}

/// One layer's down projection, read from its stacked panels.
pub struct Projection<'a> {
    pub layers: &'a stack::Layers<'a>,
    pub index: usize,
}

/// The position of down value (`hidden`, `k` of a chip's 8,192 padded features) in `Stored`.
pub fn index(hidden: usize, k: usize) -> usize {
    let h = hidden;
    [
        (4, h / 1280),
        (2, h % 1280 / 640),
        (2, h % 640 / 320),
        (2, k / 4096),
        (2, k % 4096 / 2048),
        (8, k % 2048 / 256),
        (4, h % 320 / 80),
        (80, h % 80),
        (256, k % 256),
    ]
    .iter()
    .fold(0, |index, &(size, digit)| index * size + digit)
}

impl Projection<'_> {
    /// K panel `side` (the 2,048 features `Intermediate % 4096 / 2048` of each cluster), one DMA.
    fn panel(&self, device: &mut Device, side: usize) -> Weight {
        self.layers
            .down
            .at(self.index)
            .tile::<m![Intermediate # 8192 % 4096 / 2048], 1, m![
                Depth = 1 # Depth,
                Hidden / 2560,
                Hidden % 2560 / 1280,
                Hidden % 1280 / 640,
                Hidden % 640 / 320,
                Intermediate # 8192 / 4096,
                Intermediate # 8192 % 4096 / 2048 = 1 # 2,
                Intermediate # 8192 % 2048 / 256,
                Hidden % 320 / 80,
                Hidden % 80,
                Intermediate # 8192 % 256
            ]>(side)
            .to_dm(&mut device.tdma)
    }

    pub(super) fn land(&self, device: &mut Device) -> Panel {
        Panel(self.panel(device, 0))
    }

    pub(super) fn land_right<'a>(&self, device: &mut Device, Panel(left): &'a Panel) -> Panels<'a> {
        Panels {
            left,
            right: self.panel(device, 1),
        }
    }

    /// Every wave's down projection, each followed by its cross-chip exchange and residual: a
    /// wave's exchange is issued before the next wave contracts, and its sum waits only at the
    /// residual, so the partial is in flight under the next contraction. Each wave's `residuals`
    /// rows plus its partials land in `hidden`.
    pub(super) fn sweep(
        &self,
        device: &mut Device,
        (product, scales): (&Product, &scale::Epilogue<Half, Reduced, Chips>),
        (residuals, hidden, output): (&HiddenRows, &mut HiddenRows, &mut Output),
        (waves, Panel(left)): (usize, &Panel),
    ) {
        let raw = reload(device, product, 0);
        // The second panel streams under the first down contraction, after the first reload.
        let right = self.panel(device, 1);
        let sum = reduce(device, (left, &right), raw);
        let mut raw = Raw::new();
        for _ in 0..(waves - 1).min(1) {
            reload_into(device, product, 1, &mut raw);
        }
        let crossed = cross(device, &sum);
        let mut partial = finish(scales, device, crossed);
        record(device, &partial, output, 0);
        for step in 0..waves - 1 {
            // The rows land ahead of the exchange, which is in flight under the next contraction.
            let rows: Partial = step::wave(residuals, step).to_dm(&mut device.tdma);
            let sent = residual::exchange(device, partial);
            let sum = reduce(device, (left, &right), raw);
            let mut next = Raw::new();
            for _ in 0..(waves - step - 2).min(1) {
                reload_into(device, product, step + 2, &mut next);
            }
            // The cluster halves of the next partial cross while this wave waits for its exchange.
            let crossed = cross(device, &sum);
            complete(device, (sent, &rows), (hidden, step));
            partial = finish(scales, device, crossed);
            record(device, &partial, output, step + 1);
            raw = next;
        }
        let rows: Partial = step::wave(residuals, waves - 1).to_dm(&mut device.tdma);
        let sent = residual::exchange(device, partial);
        complete(device, (sent, &rows), (hidden, waves - 1));
    }
}

/// Wave `wave`'s residual `rows` plus both ranks' partials, stored into its hidden wave.
pub(super) fn complete(
    device: &mut Device,
    (sent, rows): (
        residual::Exchanged<Half, Reduced, PartialTime, PartialPacket>,
        &Partial,
    ),
    (hidden, wave): (&mut HiddenRows, usize),
) {
    residual::finish(device, sent, rows)
        .view()
        .to_hbm_view(&mut device.tdma, step::wave_mut(hidden, wave));
}

/// The traced composition checks one wave's down partial; the step itself never reads it back.
#[cfg_attr(not(feature = "trace"), allow(unused_variables))]
pub fn record(device: &mut Device, partial: &Partial, output: &mut Output, wave: usize) {
    #[cfg(feature = "trace")]
    partial.view().to_hbm_view(
        &mut device.tdma,
        output
            .view_mut()
            .tile::<m![Waves], 1, m![Waves = 1 #{!} Waves, { Half }, { Reduced }, Rows % 8, Hidden % 80]>(
                wave,
            ),
    );
}

/// Both K panels' contractions of one wave, summed per cluster before the cluster exchange.
fn reduce(device: &mut Device, (left, right): (&Weight, &Weight), raw: Raw) -> Values {
    let input = relayout(device, &raw);
    let filled = fill(device, &input, 0);
    let low = contract(device, (left, &filled));
    let filled = fill(device, &input, 1);
    let high = contract(device, (right, &filled));
    fold(device, (low, high))
}

/// A reloaded product in the layout the register fills read.
pub(super) fn relayout(device: &mut Device, raw: &Raw) -> Activation {
    device
        .main
        .begin(raw.view())
        .fetch::<m![
            Intermediate # 8192 % 32 / 8,
            Intermediate # 8192 % 8,
            Rows % 32 / 16,
            Rows % 16 / 8
        ], Lanes>()
        .switch::<m![
            Intermediate # 8192 % 2048 / 64,
            Rows / 32,
            Rows % 32 / 16,
            Rows % 16 / 8
        ], m![
            Intermediate # 8192 % 32 / 8,
            Intermediate # 8192 % 8,
            Intermediate # 8192 % 4096 / 2048,
            Intermediate # 8192 % 64 / 32
        ]>(SwitchConfig::InterTranspose {
            slice0: 1,
            slice1: 4,
            time0: 1,
        })
        .collect::<m![
            Intermediate # 8192 % 32,
            Intermediate # 8192 % 4096 / 2048,
            Intermediate # 8192 % 64 / 32
        ], m![Rows % 8 # 32]>()
        .transpose::<m![
            Intermediate # 8192 % 32 / 8,
            Intermediate # 8192 % 4096 / 2048,
            Intermediate # 8192 % 64 / 32,
            Rows % 8
        ], m![Intermediate # 8192 % 8 # 32]>()
        .commit_trim::<m![Intermediate # 8192 % 8]>()
        .commit()
}

/// Sums the two panels' contractions of one wave.
pub(super) fn fold(device: &mut Device, (left, right): (Values, Values)) -> Values {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![Hidden / 2560, Rows % 8 / 2, Term, Rows % 2, Hidden % 80 / 8], m![Hidden % 8]>()
        .collect::<m![Hidden / 2560, Rows % 8 / 2, Term, Rows % 2, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Hidden / 2560, Rows % 8 / 2, Rows % 2, Hidden % 80 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .commit_trim::<m![Hidden % 8]>()
        .commit()
}

/// `fold` rounded once to BF16.
pub(super) fn fold_compact(device: &mut Device, (left, right): (Values, Values)) -> Folded {
    axes![Term = 2];
    device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(left.view(), right.view())
        .fetch::<m![Hidden / 2560, Rows % 8 / 2, Term, Rows % 2, Hidden % 80 / 8], m![Hidden % 8]>()
        .collect::<m![Hidden / 2560, Rows % 8 / 2, Term, Rows % 2, Hidden % 80 / 8], m![Hidden % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, m![Hidden / 2560, Rows % 8 / 2, Rows % 2, Hidden % 80 / 8]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![Hidden % 8 # 16]>()
        .commit_trim::<m![Hidden % 8]>()
        .commit()
}

fn product_wave(
    product: &Product,
    wave: usize,
) -> HbmTensorView<
    '_,
    f8e4m3,
    Chips,
    m![
        Waves = 1 # Waves,
        Rows / 32,
        Intermediate / 8,
        Rows % 32 / 16,
        Intermediate % 8,
        Rows % 16
    ],
> {
    product.view().tile::<m![Waves], 1, m![
        Waves = 1 # Waves,
        Rows / 32,
        Intermediate / 8,
        Rows % 32 / 16,
        Intermediate % 8,
        Rows % 16
    ]>(wave)
}

pub(super) fn reload(device: &mut Device, product: &Product, wave: usize) -> Raw {
    product_wave(product, wave).to_dm(&mut device.tdma)
}

pub(super) fn reload_into(device: &mut Device, product: &Product, wave: usize, raw: &mut Raw) {
    product_wave(product, wave).to_dm_view(&mut device.tdma, raw.view_mut());
}

/// Panel `half`'s share of a relaid product in the register file.
pub(super) fn fill(device: &mut Device, input: &Activation, half: usize) -> Register {
    device
        .sub
        .begin(
            input
                .view()
                .tile::<m![Intermediate # 8192 % 4096 / 2048], 1, m![
                    Intermediate # 8192 % 4096 / 2048 = 1 # 2,
                    Rows % 8,
                    Intermediate # 8192 % 64
                ]>(half),
        )
        .fetch::<m![Rows % 8, Intermediate # 8192 % 64 / 32], m![Intermediate # 8192 % 32]>()
        .switch::<Slice, m![
            Rows % 8,
            Intermediate # 8192 % 64 / 32,
            Intermediate # 8192 % 256 / 64,
            Rows / 8
        ]>(SwitchConfig::TransposedBroadcast1 {
            slice0: 32,
            slice1: 8,
        })
        .collect::<m![
            Rows % 8,
            Intermediate # 8192 % 64 / 32,
            Intermediate # 8192 % 256 / 64,
            Rows / 8
        ], m![Intermediate # 8192 % 32]>()
        .to_trf()
}

/// One panel's contraction of a filled register.
pub(super) fn contract(device: &mut Device, (weight, input): (&Weight, &Register)) -> Values {
    crate::ops::linear::Affine::new(weight.view(), ())
        .forward((
            device,
            crate::ops::layout::Dm::<Space, _> {
                tiles: crate::compose::Tiles::over(Rows::SIZE),
                value: (input,),
            },
        ))
        .value
        .transpose::<m![Hidden % 80 / 4, Hidden / 2560, Hidden % 4 / 2, Rows % 8], m![Hidden % 2 # 8]>()
        .commit_trim::<m![Hidden % 2]>()
        .commit()
}

/// A wave's partials of both clusters for the hidden half each cluster owns.
pub struct Crossed<D: Scalar = f32> {
    local: Owned<D>,
    remote: Owned<D>,
}
type Owned<D> = DmTensor<D, Chips, Cluster, Reduced, m![{ PartialTime }, { PartialPacket }]>;

/// Each cluster keeps its own hidden half in place and receives the other cluster's partial of
/// it, so only half the partial crosses; work issued before `finish` runs under the transfer.
pub(super) fn cross<D: Scalar>(
    device: &mut Device,
    partial: &DmTensor<D, Chips, Cluster, Reduced, m![Hidden / 2560, Rows % 8, Hidden % 80]>,
) -> Crossed<D> {
    let remote = partial
        .cluster_swap()
        .cluster_slice::<m![Hidden / 2560], m![{ PartialTime }, { PartialPacket }]>([0, 1])
        .to_dm(&mut device.tdma);
    let local = partial
        .asymmetric_cluster_slice::<m![Hidden / 2560], m![{ PartialTime }, { PartialPacket }]>(
            &mut device.sub,
            &[0, 1],
        );
    Crossed { local, remote }
}

/// Both clusters' partials of the hidden half each cluster owns, summed and scaled by the down
/// input and weight `scales`, rounded once to BF16 into `output`, a view of a wider buffer.
pub(super) fn finish_into<D: MaterializableScalar + FetchCast<f32>, E: M>(
    scales: &scale::Epilogue<Half, Reduced, Chips>,
    device: &mut Device,
    Crossed { local, remote }: Crossed<D>,
    output: DmTensorViewMut<'_, bf16, Chips, Half, Reduced, E>,
) {
    axes![Term = 2];
    // Two streams interleaved a packet apart leave the VE cache one vector, which reads them
    // several times slower a flit than 20-step blocks do.
    type Blocks = m![{ PartialTime } / 20, Term, { PartialTime } % 20];
    const { assert!(PartialTime::SIZE % 20 == 0) };
    // SAFETY: cluster `c` holds hidden half `c`, so naming the cluster axis by it moves no data.
    let (local, remote) = unsafe {
        (
            local.reshape::<Chips, Half, Reduced, m![{ PartialTime }, { PartialPacket }]>(),
            remote.reshape::<Chips, Half, Reduced, m![{ PartialTime }, { PartialPacket }]>(),
        )
    };
    let value = device
        .main
        .begin_interleaved::<Term, _, _, _, _, _>(local.view(), remote.view())
        .fetch::<Blocks, PartialPacket>()
        .fetch_cast::<f32>()
        .collect::<Blocks, PartialPacket>()
        .vector_init()
        .vector_intra_slice_unzip::<Term, PartialTime>()
        .vector_narrow_split::<m![{ PartialTime }, { PartialPacket } / 4], m![{ PartialPacket } % 4]>()
        .vector_fp_zip(FpBinaryOp::AddF);
    crate::ops::linear::Affine::new((), scales)
        .forward(crate::ops::layout::Dm::<Space, _> {
            tiles: crate::compose::Tiles::over(Rows::SIZE),
            value,
        })
        .value
        .vector_widen_concat::<PartialTime, PartialPacket>()
        .vector_final()
        .cast::<bf16, m![{ PartialPacket } # 16]>()
        .commit_trim::<PartialPacket>()
        .commit_view(output)
}

pub(super) fn finish<D: MaterializableScalar + FetchCast<f32>>(
    scales: &scale::Epilogue<Half, Reduced, Chips>,
    device: &mut Device,
    crossed: Crossed<D>,
) -> Partial {
    let mut partial = Partial::new();
    finish_into(scales, device, crossed, partial.view_mut());
    partial
}
