//! Rows that quantize to E4M3 at run time, each under a scale of its own, contracted against the
//! codes of a channel-scaled weight as they landed: the arithmetic of [`Token`].
//!
//! - A row's scale is its largest magnitude over E4M3's largest value, rounded once to BF16; its
//!   inputs divide by it before they round to E4M3, and each of its sums takes it and the sum's
//!   output scale before the one rounding to BF16.
//! - The feature slices a norm leaves the rows in reduce each row's largest magnitude among
//!   themselves, as the norm reduces its squares, so a row's scale and codes cost one pass over
//!   sixteen inputs a slice.
//! - The codes reach the slices of every part once for all projections of the same rows, then fill
//!   the tensor registers from data memory. A live lane group at a time rides the switch ring
//!   ([`Tokens`]) and fills the registers for every slab: a tensor-unit pass, it does not wait for
//!   the weights that transfers land meanwhile. Once the live rows are dense in the envelope, each
//!   half of it, padding and all, is stored to HBM as the feature slices hold it, landed in the
//!   slices of each part of every block and relaid row-major ([`TokenBanks`]), the second half
//!   landing while the first fills its bank and contracts: the ring would take longer than waiting
//!   for the transfers ahead and landing.
//! - A part holds whole E4M3 steps, so each contraction step takes two packets.

use super::*;
use crate::ops::linear::decode::Landed;
use crate::ops::vector::Quantization;
use crate::tensor::form::fp8::Token;

/// The codes the weights store and the rows quantize to.
type Code = f8e4m3;

/// Codes one contraction step takes: two packets.
const STEP: usize = 64;

/// The slices every row's scale reaches.
type Every = m![256];

/// The rows' E4M3 codes in the slices of each contraction part of the `In` by `Out` tiling.
type Codes<In, Out, Rows> = DmTensor<
    Code,
    m![1],
    Cluster<In, Out, Code>,
    ProjectionSlices<In, Out, Code>,
    m![Rows, { Chunked<In, Out, Code> }],
>;

/// The rows' E4M3 codes where the norm leaves the rows: sixteen inputs of each row a slice, both
/// clusters holding every row.
type Coded<In, Rows> =
    DmTensor<Code, m![1], normalization::Cluster, FeatureSlices<In>, m![Rows, In % 16]>;

/// Codes moving from the feature slices to the slices of each part, `B` of the rows at a time: in
/// HBM as the feature slices hold them, then in every block's slices of each part, a part's
/// sixteen-input runs in turn. A batch is outermost in both, so a batch of a part is one run in
/// HBM and one in each of its slices.
struct Staging<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName> {
    stored: HbmTensor<Code, m![1], m![Rows / B, In / 16, Rows % B, In % 16]>,
    landed: DmTensor<
        Code,
        m![1],
        Cluster<In, Out, Code>,
        ProjectionSlices<In, Out, Code>,
        m![Rows / B, { Chunked<In, Out, Code> } / 16, Rows % B, In % 16],
    >,
}

/// Each row's BF16 scale in every slice.
type RowScales<In, Out, Rows> = DmTensor<bf16, m![1], Cluster<In, Out, Code>, Every, m![Rows]>;

/// The rows' E4M3 codes in the slices of each contraction part of the `In` by `Out` tiling, and each
/// row's BF16 scale in every slice: staged once, they serve every projection of the same rows whose
/// tiling places each part's inputs on the same slices.
pub struct Tokens<In: AxisName, Out: AxisName, Rows: AxisName> {
    codes: Codes<In, Out, Rows>,
    scales: RowScales<In, Out, Rows>,
}

/// Half the envelope's rows: the rows one bank of the tensor registers holds.
type Half<In, Out, Rows> = Batch<In, Out, Code, Rows>;

/// The rows of an envelope, padding and all, in both banks of the tensor registers as the
/// contraction of `In` inputs into `Out` outputs reads them, half the envelope a bank, beside each
/// row's scale: filled once, they serve every slab of every congruent projection of the same rows.
pub struct TokenBanks<In: AxisName, Out: AxisName, Rows: AxisName> {
    low: Lanes<In, Out, Rows, Half<In, Out, Rows>>,
    high: Lanes<In, Out, Rows, Half<In, Out, Rows>>,
    scales: RowScales<In, Out, Rows>,
}

/// A row's scale over sixteen inputs a feature slice, the feature slices reducing it among
/// themselves, BF16 and four rows a packet.
type Scales<In, Rows> = Quantization<
    m![Rows, In % 16 / 8],
    m![Rows, In % 16 / 4],
    In,
    m![Rows],
    Every,
    m![Rows / 4],
    m![Rows % 4],
>;

/// A batch of `B` of the `Rows` in the tensor registers as E4M3 lanes of the contraction's slices.
type Lanes<In, Out, Rows, B> = TrfTensor<
    Code,
    m![1],
    Cluster<In, Out, Code>,
    ProjectionSlices<In, Out, Code>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Count<Rows, B> }, { Chunked<In, Out, Code> }],
>;

/// The cells of `B` of the `Rows`, chosen from the batches.
type Batched<Rows, B, Inner> = m![Rows / B = 1 # (Rows / B), Rows % B, { Inner }];
/// `B` of the `Rows`, chosen from the batches.
type Chosen<Rows, B> = m![Rows / B = 1 # (Rows / B), Rows % B];
/// The cells of `B` of the `Rows`, to write.
type BatchedMut<Rows, B, Inner> = m![Rows / B = 1 #{!} (Rows / B), Rows % B, { Inner }];

impl<In: AxisName, Out: AxisName> Contraction<In, Out, Code> {
    /// The live lane groups of `input`, each row quantized under its own scale, their codes in the
    /// slices of each part.
    pub(crate) fn tokens<Rows: AxisName>(
        device: &mut Device,
        input: &Live<Spread<In, Rows>, Rows>,
    ) -> Tokens<In, Out, Rows> {
        let (coded, scales) = Self::quantized(device, input.value());
        let mut codes = DmTensor::new();
        let (first, count) = input.groups::<axis::Lanes<Rows>>();
        for at in 0..count {
            Self::part::<Rows, axis::Lanes<Rows>>(device, &coded, &mut codes, first + at);
        }
        Tokens { codes, scales }
    }

    /// Every row of `input`, padding and all, each quantized under its own scale, against every
    /// slab of `landed`, each output rounded once to BF16, as [`Self::kept`] contracts BF16 rows:
    /// the second half of the envelope lands while the first fills its bank of the registers and
    /// contracts the first slab, and both banks stay filled for every congruent projection of the
    /// same rows.
    pub(crate) fn token_kept<Rows: AxisName>(
        device: &mut Device,
        landed: &Landed<In, Out, Token>,
        input: &Spread<In, Rows>,
    ) -> (Result<In, Out, Code, Rows>, TokenBanks<In, Out, Rows>) {
        const {
            assert!(
                Rows::SIZE == 2 * <Half<In, Out, Rows> as AxisName>::SIZE,
                "a bank holds half the envelope"
            )
        };
        let (coded, scales) = Self::quantized(device, input);
        let mut staging = Staging::<In, Out, Rows, Half<In, Out, Rows>>::new();
        let mut codes = DmTensor::new();
        let mut result = DmTensor::new();
        staging.land(device, &coded, 0);
        staging.relay(device, &mut codes, 0);
        staging.land(device, &coded, 1);
        let mut bank = |device: &mut Device, codes: &Codes<In, Out, Rows>, at: usize| {
            let lanes = Self::fill::<Rows, Half<In, Out, Rows>>(device, codes, at);
            Self::contracted::<Rows, Half<In, Out, Rows>, Out>(
                device,
                landed,
                (&lanes, &scales),
                &mut result,
                (0, at),
            );
            lanes
        };
        let low = bank(&mut *device, &codes, 0);
        staging.relay(device, &mut codes, 1);
        let high = bank(&mut *device, &codes, 1);
        #[unroll]
        for slab in 1..<Passes<In, Out, Code> as M>::SIZE {
            Self::contracted::<Rows, Half<In, Out, Rows>, Out>(
                device,
                landed,
                (&low, &scales),
                &mut result,
                (slab, 0),
            );
            Self::contracted::<Rows, Half<In, Out, Rows>, Out>(
                device,
                landed,
                (&high, &scales),
                &mut result,
                (slab, 1),
            );
        }
        (result, TokenBanks { low, high, scales })
    }

    /// The rows `banks` hold, filled as the congruent tiling of `F` outputs reads them, against
    /// every slab of `landed`, each output rounded once to BF16.
    pub(crate) fn token_banked<Rows: AxisName, F: AxisName>(
        device: &mut Device,
        landed: &Landed<In, Out, Token>,
        banks: &TokenBanks<In, F, Rows>,
    ) -> Result<In, Out, Code, Rows> {
        const {
            assert!(
                Self::congruent::<F>(),
                "the tilings place each part's inputs on the same slices"
            )
        };
        let mut result = DmTensor::new();
        #[unroll]
        for slab in 0..<Passes<In, Out, Code> as M>::SIZE {
            Self::contracted::<Rows, Half<In, F, Rows>, F>(
                device,
                landed,
                (&banks.low, &banks.scales),
                &mut result,
                (slab, 0),
            );
            Self::contracted::<Rows, Half<In, F, Rows>, F>(
                device,
                landed,
                (&banks.high, &banks.scales),
                &mut result,
                (slab, 1),
            );
        }
        result
    }

    /// The rows `tokens` staged for the congruent tiling of `F` outputs, a live lane group at a
    /// time, against every slab of `landed`, each output rounded once to BF16: each group fills the
    /// tensor registers once for every slab.
    pub(crate) fn tokened<Rows: AxisName, F: AxisName>(
        device: &mut Device,
        landed: &Landed<In, Out, Token>,
        tokens: &Live<Tokens<In, F, Rows>, Rows>,
    ) -> Result<In, Out, Code, Rows> {
        const {
            assert!(
                Self::congruent::<F>(),
                "the tilings place each part's inputs on the same slices"
            )
        };
        let staged = tokens.value();
        let mut result = DmTensor::new();
        let (first, count) = tokens.groups::<axis::Lanes<Rows>>();
        for at in 0..count {
            let group = first + at;
            let lanes = Contraction::<In, F, Code>::fill::<Rows, axis::Lanes<Rows>>(
                device,
                &staged.codes,
                group,
            );
            #[unroll]
            for slab in 0..<Passes<In, Out, Code> as M>::SIZE {
                Self::contracted::<Rows, axis::Lanes<Rows>, F>(
                    device,
                    landed,
                    (&lanes, &staged.scales),
                    &mut result,
                    (slab, group),
                );
            }
        }
        result
    }

    /// Each row of `input` quantized under its own scale where the norm leaves it, and the scales
    /// in every slice under the contraction's names.
    fn quantized<Rows: AxisName>(
        device: &mut Device,
        input: &Spread<In, Rows>,
    ) -> (Coded<In, Rows>, RowScales<In, Out, Rows>) {
        const {
            assert!(
                normalization::Chunk::<In>::SIZE == 16,
                "a feature slice holds sixteen inputs of a row"
            )
        };
        let rows = input.view();
        let scales: DmTensor<bf16, m![1], normalization::Cluster, Every, m![Rows]> =
            Scales::<In, Rows>::scales(&mut device.main, rows.clone()).commit();
        // SAFETY: the reduction leaves each row's scale in every slice.
        let local = unsafe {
            scales
                .view()
                .reshape::<m![1], normalization::Cluster, FeatureSlices<In>, m![Rows]>()
        };
        let scale: VrfTensor<f32, m![1], normalization::Cluster, FeatureSlices<In>, m![Rows]> =
            device
                .sub
                .begin(local)
                .fetch::<Layout<Rows>::Groups, Layout<Rows>::Lanes>()
                .fetch_cast::<f32>()
                .collect::<Layout<Rows>::Groups, Layout<Rows>::Lanes>()
                .to_vrf();
        let coded = Scales::<In, Rows>::rows(&mut device.main, rows, &scale).commit();
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        (coded, unsafe { scales.reshape() })
    }

    /// Batch `at` of `B` of the coded rows, from their feature slices into the slices of each part
    /// over the switch ring: each slice's sixteen inputs ride a packet of their own.
    fn part<Rows: AxisName, B: AxisName>(
        device: &mut Device,
        coded: &Coded<In, Rows>,
        codes: &mut Codes<In, Out, Rows>,
        at: usize,
    ) {
        // SAFETY: both physical clusters hold every row; only the logical cluster axis changes.
        let coded = unsafe {
            coded
                .view()
                .reshape::<m![1], Cluster<In, Out, Code>, FeatureSlices<In>, m![Rows, In % 16]>()
        };
        device
            .sub
            .begin(coded.tile::<m![Rows / B], 1, Batched<Rows, B, m![In % 16]>>(at))
            .fetch::<m![Rows % B], m![In % 16]>()
            .switch::<ProjectionSlices<In, Out, Code>, m![
                Rows % B,
                { Chunked<In, Out, Code> } / 16
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![Rows % B, { Chunked<In, Out, Code> } / 16], m![In % 16 # 32]>()
            .commit_trim::<m![In % 16]>()
            .commit_view(
                codes
                    .view_mut()
                    .tile::<m![Rows / B], 1, BatchedMut<Rows, B, Chunked<In, Out, Code>>>(at),
            );
    }

    /// Batch `at` of `B` of the staged `codes`, from data memory into the tensor registers as the
    /// contraction reads them.
    fn fill<Rows: AxisName, B: AxisName>(
        device: &mut Device,
        codes: &Codes<In, Out, Rows>,
        at: usize,
    ) -> Lanes<In, Out, Rows, B> {
        device
            .sub
            .begin(
                codes
                    .view()
                    .tile::<m![Rows / B], 1, Batched<Rows, B, Chunked<In, Out, Code>>>(at),
            )
            .fetch::<m![
                { Layout<Rows>::Lanes },
                { Count<Rows, B> },
                { Chunked<In, Out, Code> } / 32
            ], m![In % 32]>()
            .collect::<m![
                { Layout<Rows>::Lanes },
                { Count<Rows, B> },
                { Chunked<In, Out, Code> } / 32
            ], m![In % 32]>()
            .to_trf()
    }

    /// Batch `at` of `B` of the rows in `lanes`, filled as the congruent tiling of `F` outputs reads
    /// them, against slab `slab` of `landed`: the codes contract a step of two packets at a time,
    /// take their rows' and their outputs' scales once the parts meet, and round once into the
    /// batch's cells of `result`.
    fn contracted<Rows: AxisName, B: AxisName, F: AxisName>(
        device: &mut Device,
        landed: &Landed<In, Out, Token>,
        (lanes, scales): (
            &Lanes<In, F, Rows, B>,
            &DmTensor<bf16, m![1], Cluster<In, F, Code>, Every, m![Rows]>,
        ),
        result: &mut Table<bf16, In, Out, Code, Rows, m![1]>,
        (slab, at): (usize, usize),
    ) {
        let scales = scales.view().tile::<m![Rows / B], 1, Chosen<Rows, B>>(at);
        // SAFETY: every slice holds the row scales; the slices that keep the sums read them.
        let scales = unsafe {
            scales.reshape::<m![1], Cluster<In, F, Code>, Reduced<In, F, Code>, Chosen<Rows, B>>()
        };
        let row: VrfTensor<
            f32,
            m![1],
            Cluster<In, F, Code>,
            Reduced<In, F, Code>,
            m![{ Count<Rows, B> }, { Layout<Rows>::Lanes }],
        > = device
            .sub
            .begin(scales)
            .fetch::<m![{ Count<Rows, B> }], m![{ Layout<Rows>::Lanes }]>()
            .fetch_cast::<f32>()
            .collect::<m![{ Count<Rows, B> }], m![{ Layout<Rows>::Lanes }]>()
            .to_vrf();
        // SAFETY: congruent tilings order a slice's codes and scales alike under either tiling's
        // names; a block's scales sit in its first part's slices, which keep the sums.
        let (codes, outputs) = unsafe {
            (
                landed.codes(slab).reshape::<
                    m![1],
                    Cluster<In, F, Code>,
                    ProjectionSlices<In, F, Code>,
                    m![
                        { Slab<In, Out, Code> },
                        Out % ({ Block<In, Out, Code> }),
                        { Chunked<In, F, Code> }
                    ],
                >(),
                landed.scales(slab).reshape::<
                    m![1],
                    Cluster<In, F, Code>,
                    Reduced<In, F, Code>,
                    m![{ Slab<In, Out, Code> }, Out % ({ Block<In, Out, Code> }), 1 # 8],
                >(),
            )
        };
        let output: VrfTensor<
            f32,
            m![1],
            Cluster<In, F, Code>,
            Reduced<In, F, Code>,
            m![Out % ({ Block<In, Out, Code> }), 1 # 8],
        > = device
            .sub
            .begin(outputs)
            .fetch::<m![Out % ({ Block<In, Out, Code> })], m![1 # 8]>()
            .collect::<m![Out % ({ Block<In, Out, Code> })], m![1 # 8]>()
            .to_vrf();
        const {
            assert!(
                <Chunked<In, F, Code> as M>::SIZE.is_multiple_of(STEP),
                "a part is whole E4M3 steps"
            )
        };
        let scaled = device
            .main
            .begin(codes)
            .fetch::<m![
                Out % ({ Block<In, Out, Code> }),
                { Chunked<In, F, Code> } / 64
            ], m![In % 64]>()
            .collect::<m![
                Out % ({ Block<In, Out, Code> }),
                { Chunked<In, F, Code> } / 32
            ], m![In % 32]>()
            .contract_outer::<m![
                Out % ({ Block<In, Out, Code> }),
                { Chunked<In, F, Code> } / 64,
                { Count<Rows, B> }
            ], m![In % 64], _, _, _>(lanes)
            .contract_packet::<m![1]>()
            .contract_time::<m![Out % ({ Block<In, Out, Code> }), { Count<Rows, B> }]>()
            .contract_lane::<
                m![Out % ({ Block<In, Out, Code> }), { Count<Rows, B> }],
                m![{ Layout<Rows>::Lanes } # 8],
            >(LaneMode::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<
                Reduced<In, F, Code>,
                m![Out % ({ Block<In, Out, Code> }), { Count<Rows, B> }],
            >(InterSliceReduceOpF32::Add)
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_narrow_split::<m![
                Out % ({ Block<In, Out, Code> }),
                { Count<Rows, B> },
                { Layout<Rows>::Lanes } # 8 / 4
            ], m![{ Layout<Rows>::Lanes } # 8 % 4]>()
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &row)
            .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &output)
            .vector_widen_concat::<
                m![Out % ({ Block<In, Out, Code> }), { Count<Rows, B> }],
                m![{ Layout<Rows>::Lanes } # 8],
            >()
            .vector_final();
        let cells = Self::cells::<Rows, B, m![1], bf16>(result, slab, at);
        // SAFETY: as the slab's, the cells' clusters and slices under the congruent tiling's names.
        let cells = unsafe {
            cells.reshape::<
                m![1],
                Cluster<In, F, Code>,
                Reduced<In, F, Code>,
                Cells<bf16, In, Out, Code, Rows, B>,
            >()
        };
        <bf16 as Rounding>::round::<
            _,
            _,
            _,
            _,
            Layout<Rows>::Lanes,
            m![
                { <bf16 as Rounding>::Packets<m![Out % ({ Block<In, Out, Code> })]> },
                { Count<Rows, B> },
                { Layout<Rows>::Lanes }
            ],
            <bf16 as Rounding>::Packet<Out>,
            _,
        >(scaled)
        .commit_view(cells);
    }
}

impl<In: AxisName, Out: AxisName, Rows: AxisName, B: AxisName> Staging<In, Out, Rows, B> {
    fn new() -> Self {
        Self {
            stored: HbmTensor::new(),
            landed: DmTensor::new(),
        }
    }

    /// Batch `at` of the coded rows into the slices of each part of every block, through HBM:
    /// each feature slice stores the batch's rows as one run, and each part's slice lands its
    /// sixteen-input runs, one run in HBM and in the slice.
    fn land(&mut self, device: &mut Device, coded: &Coded<In, Rows>, at: usize) {
        coded
            .view()
            .tile::<m![Rows / B], 1, Batched<Rows, B, m![In % 16]>>(at)
            .to_hbm_view(
                &mut device.tdma,
                self.stored.view_mut().tile::<m![Rows / B], 1, m![
                    Rows / B = 1 #{!} (Rows / B), In / 16, Rows % B, In % 16
                ]>(at),
            );
        self.stored
            .view()
            .tile::<m![Rows / B], 1, m![Rows / B = 1 # (Rows / B), In / 16, Rows % B, In % 16]>(at)
            .to_dm_view(
                &mut device.tdma,
                self.landed.view_mut().tile::<m![Rows / B], 1, m![
                    Rows / B = 1 #{!} (Rows / B),
                    { Chunked<In, Out, Code> } / 16,
                    Rows % B,
                    In % 16
                ]>(at),
            );
    }

    /// Batch `at` of the landed codes relaid row-major in `codes`, as the registers read them.
    fn relay(&self, device: &mut Device, codes: &mut Codes<In, Out, Rows>, at: usize) {
        device
            .main
            .begin(self.landed.view().tile::<m![Rows / B], 1, m![
                Rows / B = 1 # (Rows / B),
                { Chunked<In, Out, Code> } / 16,
                Rows % B,
                In % 16
            ]>(at))
            .fetch::<m![{ Chunked<In, Out, Code> } / 16, Rows % B], m![In % 16]>()
            .collect::<m![{ Chunked<In, Out, Code> } / 16, Rows % B], m![In % 16 # 32]>()
            .commit_trim::<m![In % 16]>()
            .commit_view(
                codes
                    .view_mut()
                    .tile::<m![Rows / B], 1, BatchedMut<Rows, B, Chunked<In, Out, Code>>>(at),
            );
    }
}
