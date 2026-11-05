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

/// The absolute hidden feature and head feature of a `Stored` position on chip `chip`.
fn place(chip: usize, index: usize) -> (usize, usize) {
    let lanes_tile = 8 * 1280;
    let (part, rest) = (
        index / (2 * 256 * lanes_tile),
        index % (2 * 256 * lanes_tile),
    );
    let (cluster, rest) = (rest / (256 * lanes_tile), rest % (256 * lanes_tile));
    let (slice, rest) = (rest / lanes_tile, rest % lanes_tile);
    let (lane, tile) = (rest / 1280, rest % 1280);
    let (m, h8, group) = (slice / 128, slice % 128 / 16, slice % 16);
    let (w4, hg, w32) = (tile / 320, tile % 320 / 32, tile % 32);
    let hidden = 2560 * chip + 1280 * m + 640 * cluster + 80 * h8 + 8 * hg + lane;
    let owner = if part == 0 { chip } else { 1 - chip };
    (hidden, owner * 2048 + 128 * group + 32 * w4 + w32)
}

/// Packs both chips' TP2 O shards (`output::Stored`, chip `s` holding its heads' 2,048 features
/// by all 5,120 hidden rows) into each chip's own-half weight (`Stored`).
pub fn pack<T: Copy>(shards: [&[T]; 2]) -> [Vec<T>; 2] {
    // TP2 order: [Hidden / 2560, Hidden % 2560 / 80, Width / 256, Hidden % 8, Width % 256 / 32,
    // Hidden % 80 / 8, Width % 32].
    let at = |hidden: usize, width: usize| {
        (((((hidden / 2560 * 32 + hidden % 2560 / 80) * 8 + width / 256) * 8 + hidden % 8) * 8
            + width % 256 / 32)
            * 10
            + hidden % 80 / 8)
            * 32
            + width % 32
    };
    std::array::from_fn(|chip| {
        (0..Stored::SIZE)
            .map(|index| {
                let (hidden, feature) = place(chip, index);
                shards[feature / 2048][at(hidden, feature % 2048)]
            })
            .collect()
    })
}

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

/// One part in the register file with the scales its epilogue reads.
pub struct FilledPart {
    register: Register,
    epilogue: scale::Epilogue<Cluster, Reduced, Chips>,
}

impl Weight {
    /// Part `PART` alone in the register file (0: own heads, 1: the peer's).
    pub fn fill_part<const PART: usize>(&self, device: &mut Device) -> FilledPart {
        FilledPart {
            register: fill_part(device, &self.weight[PART]),
            epilogue: scale::Epilogue {
                input: self.scales.get(device, 0),
                weight: self.scales.get(device, 1),
            },
        }
    }
}

impl FilledPart {
    /// This part's contraction of one wave's rows, scaled, rounded to BF16.
    pub fn project(&self, device: &mut Device, rows: &Rows8) -> Output {
        crate::ops::linear::Affine::new(&self.register, &self.epilogue)
            .forward((
                device,
                crate::ops::layout::Dm::<Space, _> {
                    tiles: crate::compose::Tiles::over(Rows::SIZE),
                    value: rows,
                },
            ))
            .value
    }
}

/// Both parts' outputs of one wave summed and rounded again, into `output`.
pub fn sum_into<E: M>(
    device: &mut Device,
    (first, second): (&Output, &Output),
    output: DmTensorViewMut<'_, bf16, Chips, Cluster, Reduced, E>,
) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    type Parts = HbmTensor<f8e4m3, Chips, m![Part, Rows / 4, Width / 128, Rows % 4, Width % 128]>;

    #[device(chip = 2)]
    fn project(
        device: &mut Device,
        rows: &Parts,
        (weight, scales): (
            &HbmTensor<f8e4m3, Chips, Stored>,
            &HbmTensor<f32, Chips, scale::Packed>,
        ),
    ) -> HbmTensor<
        bf16,
        Chips,
        m![
            Hidden % 2560 / 1280,
            Hidden % 640 / 80,
            Rows / 4,
            Hidden % 1280 / 640,
            Rows % 4,
            Hidden % 80
        ],
    > {
        let part = |part: usize| {
            rows.view()
                .tile::<m![Part], 1, m![Part = 1 # 2, Rows / 4, Width / 128, Rows % 4, Width % 128]>(part)
        };
        let own: Rows8 = part(0).to_dm(&mut device.tdma);
        let peer: Rows8 = part(1).to_dm(&mut device.tdma);
        let mut landed = Weight::new();
        landed.land_from(device, (weight, scales.view()));
        let filled = landed.fill(device);
        let mut out = Output::new();
        filled.project_into(device, (&own, &peer), out.view_mut());
        out.to_hbm(&mut device.tdma)
    }

    /// Chip `c` cluster `k` outputs hidden features `2560 c + 1280 m + 640 k ..` of every head's
    /// rows (own heads in part 0) by the full O weight.
    #[tokio::test(flavor = "current_thread")]
    async fn projects_own_half() {
        // Given: small E4M3 codes (|x| <= 2), the full O weight [5120, 4096] and each chip's rows
        // with its own heads in part 0.
        let code = |index: usize| -> u8 {
            let mantissa = (index * 7 + index / 13) % 8;
            let exponent = 5 + (index / 3) % 4;
            let sign = (index * 5 / 3) % 2;
            u8::try_from((sign << 7) | (exponent << 3) | mantissa).unwrap()
        };
        let value = |code: u8| -> f64 {
            let (sign, exponent, mantissa) = (code >> 7, (code >> 3) & 15, code & 7);
            let magnitude = (1.0 + f64::from(mantissa) / 8.0) * 2f64.powi(i32::from(exponent) - 7);
            if sign == 1 { -magnitude } else { magnitude }
        };
        let full: Vec<u8> = (0..5120 * 4096).map(|index| code(index * 3 + 1)).collect();
        let absolute: Vec<u8> = (0..64 * 4096).map(|index| code(index * 11 + 5)).collect();
        let mut rows = Vec::with_capacity(2 * 2 * 64 * 2048);
        for chip in 0..2 {
            for part in 0..2 {
                let owner = if part == 0 { chip } else { 1 - chip };
                for r4 in 0..16 {
                    for group in 0..16 {
                        for row in 0..4 {
                            for feature in 0..128 {
                                rows.push(
                                    absolute[(r4 * 4 + row) * 4096
                                        + owner * 2048
                                        + group * 128
                                        + feature],
                                );
                            }
                        }
                    }
                }
            }
        }
        let tp2: Vec<Vec<u8>> = (0..2)
            .map(|shard| {
                let mut out = vec![0; 5120 * Width::SIZE];
                for hidden in 0..5120 {
                    for width in 0..Width::SIZE {
                        let at =
                            (((((hidden / 2560 * 32 + hidden % 2560 / 80) * 8 + width / 256) * 8
                                + hidden % 8)
                                * 8
                                + width % 256 / 32)
                                * 10
                                + hidden % 80 / 8)
                                * 32
                                + width % 32;
                        out[at] = full[hidden * 4096 + shard * 2048 + width];
                    }
                }
                out
            })
            .collect();
        let packed = pack([&tp2[0], &tp2[1]]);
        let mut device = Device::new(project.topology()).unwrap();
        let rows_hbm = HostTensor::<
            f8e4m3,
            m![
                { Chips },
                Part,
                Rows / 4,
                Width / 128,
                Rows % 4,
                Width % 128
            ],
        >::from_buf(rows)
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let weight = HostTensor::<f8e4m3, m![{ Chips }, { Stored }]>::from_buf(
            [packed[0].clone(), packed[1].clone()].concat(),
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();
        let scales = HostTensor::<f32, m![{ Chips }, { scale::Packed }]>::from_vec(
            vec![1.0f32; 2 * scale::Packed::SIZE],
        )
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();

        // When
        let actual = launch(project, (&mut device, &rows_hbm, (&weight, &scales)))
            .await
            .unwrap()
            .to_host::<m![
                { Chips },
                Hidden % 2560 / 1280,
                Hidden % 640 / 80,
                Rows / 4,
                Hidden % 1280 / 640,
                Rows % 4,
                Hidden % 80
            ]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();

        // Then
        for chip in 0..2 {
            let (mut error, mut energy) = (0.0, 0.0);
            for (offset, got) in actual[chip * 64 * 2560..(chip + 1) * 64 * 2560]
                .iter()
                .enumerate()
            {
                let (m, rest) = (
                    offset / (8 * 16 * 2 * 4 * 80),
                    offset % (8 * 16 * 2 * 4 * 80),
                );
                let (h8, rest) = (rest / (16 * 2 * 4 * 80), rest % (16 * 2 * 4 * 80));
                let (r4, rest) = (rest / (2 * 4 * 80), rest % (2 * 4 * 80));
                let (k, rest) = (rest / (4 * 80), rest % (4 * 80));
                let (row, j) = (r4 * 4 + rest / 80, rest % 80);
                let hidden = 2560 * chip + 1280 * m + 640 * k + 80 * h8 + j;
                let expected: f64 = (0..4096)
                    .map(|feature| {
                        value(absolute[row * 4096 + feature]) * value(full[hidden * 4096 + feature])
                    })
                    .sum();
                error += (f64::from(got.to_f32()) - expected).powi(2);
                energy += expected.powi(2);
            }
            let relative = (error / energy).sqrt();
            assert!(relative < 1e-2, "chip={chip} rel_l2={relative}");
        }
    }
}
