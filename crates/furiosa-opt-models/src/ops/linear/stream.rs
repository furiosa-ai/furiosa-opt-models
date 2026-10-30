//! A projection onto more outputs than data memory holds, such as a vocabulary: each slice keeps
//! one pass of its block of weight rows at a time, and each pass issues the next pass's weights
//! before its own contraction, so the weights stream under the compute.
//!
//! - A pass's logits, each plus its bias, go to a consumer while the pass is live: argmax's
//!   candidates, or the rows' scores in HBM.
//! - Every row of the input's envelope is projected. Candidates stage the rows' lanes once a
//!   call; scores stage a 32-row tile's lanes each pass, the most a transposition takes.

use crate::Op;
use crate::ops::index::gather;
use crate::ops::linear::kernel::{self, Unrounded};
use crate::ops::linear::{Affine, Groups};
use crate::ops::normalization::{self, FeatureSlices, Spread};
use crate::ops::reduction::Candidates;
use crate::specialize::vocabulary;
use crate::tensor::{axis::Tile, mapping::Rows as Layout, parallel::Chip};
use core::marker::PhantomData;
use furiosa_opt_std::prelude::*;
use std::borrow::Borrow;

/// A slice's vocabulary rows one pass holds: the most rows, at most `vocabulary::ROWS` and whole
/// multiples of `vocabulary::STEP`, that split the slice's `Block` of rows into passes.
#[derive(Clone, Debug)]
pub(crate) struct Pass<Block>(PhantomData<Block>);

impl<Block: M + 'static> AxisName for Pass<Block> {
    const NAME: Ident = Ident::new("Pass");

    const SIZE: usize = vocabulary::pass(<Block as M>::SIZE, vocabulary::ROWS, vocabulary::STEP);
}

pub(crate) type PassRows<Hidden, V> = Symbol<Pass<Block<Hidden, V>>>;

/// The hidden features each slice contracts, its block of every hidden row.
#[derive(Clone, Debug)]
pub(crate) struct Contraction<Hidden>(PhantomData<Hidden>);

impl<Hidden: AxisName> AxisName for Contraction<Hidden> {
    const NAME: Ident = Ident::new("Contraction");

    const SIZE: usize = vocabulary::chunk(Hidden::SIZE);
}

type Chunk<Hidden> = Symbol<Contraction<Hidden>>;

/// The features of each input row that one slice of a [`Spread`] holds.
type Feature<Hidden> = Symbol<normalization::Chunk<Hidden>>;

/// The slices sharing one row's contraction, `Hidden / chunk`.
type Parts<Hidden> = m![Hidden / ({ Chunk<Hidden> })];
/// The vocabulary rows one slice writes over every pass, `V * parts / 512`: its block of the
/// rows, `PassRows` a pass, since the blocks fill the `256 / parts` slots of both clusters.
pub(crate) type Block<Hidden, V> = m![(V, { Parts<Hidden> }) / 512];
/// The passes a slice's block takes.
pub(crate) type Passes<Hidden, V> = m![{ Block<Hidden, V> } / ({ PassRows<Hidden, V> })];

pub(crate) type Cluster<V> = m![V / (V / 2)];
type Slice<Hidden, V> =
    m![V % (V / 2) / ({ Block<Hidden, V> }), 1 # ({ m![1] }), { Parts<Hidden> }];
pub(crate) type Reduced<Hidden, V> = m![
    V % (V / 2) / ({ Block<Hidden, V> }),
    1 # ({ m![1] }),
    { m![1 # ({ Parts<Hidden> })] }
];

/// A vocabulary row's row groups, padded by one: a 512-byte step would hit the same four of a
/// slice's sixteen banks each cycle, which the emitter serializes against the weight stream.
type PaddedGroups = m![17];

/// One pass's logits of every row, in its vocabulary block's reduced slice.
pub(crate) type Logits<Hidden, V, Rows, Chips = Chip> = DmTensor<
    f32,
    Chips,
    Cluster<V>,
    Reduced<Hidden, V>,
    m![
        V % ({ PassRows<Hidden, V> }),
        { Groups<Rows> } # ({ PaddedGroups }),
        { Layout<Rows>::Lanes } # 8
    ],
>;

type Weights<Chips, Hidden, V> = DmTensor<
    bf16,
    Chips,
    Cluster<V>,
    Slice<Hidden, V>,
    m![
        V % ({ PassRows<Hidden, V> }),
        Hidden % ({ Chunk<Hidden> })
    ],
>;

/// `Count` lane groups of the rows as the tensor-register lanes of every slice, each slice the
/// rows' features in its chunk.
type Lanes<Chips, Hidden, V, Rows, Count> = TrfTensor<
    bf16,
    Chips,
    Cluster<V>,
    Slice<Hidden, V>,
    m![{ Layout<Rows>::Lanes }],
    m![{ Count }, Hidden % ({ Chunk<Hidden> })],
>;

/// A tile of at most 32 rows, the most lanes a transposition of the logits takes.
type RowTile<Rows> = Symbol<Tile<Rows, 32>>;

/// One per-vocabulary vector, every pass of each slice's block resident at once.
pub(crate) type Columns<Hidden, V, D, Chips = Chip> = DmTensor<
    D,
    Chips,
    Cluster<V>,
    Reduced<Hidden, V>,
    m![
        V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> }),
        V % ({ PassRows<Hidden, V> })
    ],
>;
/// One pass of a resident per-vocabulary vector.
pub(crate) type Column<Hidden, V> = m![
    1 # (V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })),
    V % ({ PassRows<Hidden, V> })
];

/// Every row's logits into `candidates`, pass by pass while each pass is live.
impl<'c, Hidden, V, Rows, Matrix, Offset, Chips: M + 'static>
    Op<(
        &Spread<Hidden, Rows, Chips>,
        &mut Candidates<'c, Hidden, V, Rows, Chips>,
    )> for Affine<Matrix, Offset>
where
    Hidden: AxisName,
    V: AxisName,
    Rows: AxisName,
    Matrix: Borrow<HbmTensor<bf16, Chips, m![V, Hidden]>>,
    Offset: Borrow<Columns<Hidden, V, f32, Chips>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, candidates): (
            &Spread<Hidden, Rows, Chips>,
            &mut Candidates<'c, Hidden, V, Rows, Chips>,
        ),
    ) {
        let bias = self.bias.borrow();
        let lanes = stage::<Hidden, V, Rows, Groups<Rows>, _, _>(device, input.view());
        stream(device, self.weight.borrow(), |device, weight, pass| {
            let logits: Logits<Hidden, V, Rows, Chips> =
                biased(device, (weight, &lanes, bias), pass, |sums| {
                    sums.commit_trim::<m![{ Layout<Rows>::Lanes } # 8]>()
                        .commit()
                });
            candidates.push(device, &logits, pass);
        });
    }
}

/// Every row's scores into `scores`, a pass's columns while the pass is live.
impl<Hidden, V, Rows, Matrix, Offset, Chips: M + 'static>
    Op<(
        &Spread<Hidden, Rows, Chips>,
        &mut HbmTensor<f32, Chips, m![Rows, V]>,
    )> for Affine<Matrix, Offset>
where
    Hidden: AxisName,
    V: AxisName,
    Rows: AxisName,
    Matrix: Borrow<HbmTensor<bf16, Chips, m![V, Hidden]>>,
    Offset: Borrow<Columns<Hidden, V, f32, Chips>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        (input, scores): (
            &Spread<Hidden, Rows, Chips>,
            &mut HbmTensor<f32, Chips, m![Rows, V]>,
        ),
    ) {
        let bias = self.bias.borrow();
        stream(device, self.weight.borrow(), |device, weight, pass| {
            for tile in 0..<m![Rows / ({ RowTile<Rows> })] as M>::SIZE {
                let input = input.view().tile::<m![Rows / ({ RowTile<Rows> })], 1, m![
                    Rows / ({ RowTile<Rows> }) = 1 # (Rows / ({ RowTile<Rows> })),
                    Rows % ({ RowTile<Rows> }),
                    Hidden % ({ Feature<Hidden> })
                ]>(tile);
                let lanes = stage::<Hidden, V, Rows, m![Rows % ({ RowTile<Rows> }) / 8], _, _>(
                    &mut *device,
                    input,
                );
                let logits: DmTensor<
                    f32,
                    Chips,
                    Cluster<V>,
                    Reduced<Hidden, V>,
                    m![Rows % ({ RowTile<Rows> }), V % ({ PassRows<Hidden, V> })],
                > = biased(&mut *device, (weight, &lanes, bias), pass, |sums| {
                    Unrounded::tail::<
                        _,
                        _,
                        _,
                        _,
                        m![V % ({ PassRows<Hidden, V> }) / 2, Rows % ({ RowTile<Rows> })],
                        m![V % 2],
                        _,
                    >(sums)
                    .commit()
                });
                logits.view().to_hbm_view(
                    &mut device.tdma,
                    scores
                        .view_mut()
                        .tile::<m![V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })], 1, m![
                            Rows,
                            V / ({ Block<Hidden, V> }),
                            1 #{!} ({ Passes<Hidden, V> }),
                            V % ({ PassRows<Hidden, V> })
                        ]>(pass)
                        .tile::<m![Rows / ({ RowTile<Rows> })], 1, m![
                            1 #{!} (Rows / ({ RowTile<Rows> })),
                            Rows % ({ RowTile<Rows> }),
                            V / ({ Block<Hidden, V> }),
                            1 #{!} ({ Passes<Hidden, V> }),
                            V % ({ PassRows<Hidden, V> })
                        ]>(tile),
                );
            }
        });
    }
}

/// Each row's scores of the padded vocabulary, as a borrowed tree's head projects them; every row
/// of the envelope is projected.
impl<'w, H: AxisName, V: AxisName, R: AxisName> Op<&mut gather::Rows<'_, bf16, H, R>>
    for Affine<&'w HbmTensor<bf16, Chip, m![V, H]>, &'w HbmTensor<f32, Chip, m![V]>>
{
    type Output = HbmTensor<f32, Chip, m![R, V]>;

    fn forward(
        &self,
        device: &mut Device,
        input: &mut gather::Rows<'_, bf16, H, R>,
    ) -> Self::Output {
        let hidden = ToDm::<H>(PhantomData).forward(&mut *device, &*input);
        let bias = ToDm::<H>(PhantomData).forward(&mut *device, self.bias);
        let mut scores = HbmTensor::new();
        Affine::new(self.weight, bias).forward(device, (&hidden, &mut scores));
        scores
    }
}

/// Writes a projection's operands in DM: a per-vocabulary vector as columns, or a call's live rows
/// spread over their feature slices, the later rows unspecified.
pub(crate) struct ToDm<Hidden>(pub(crate) PhantomData<Hidden>);

impl<Hidden: AxisName, V: AxisName, D: MaterializableScalar, Chips: M + 'static>
    Op<&HbmTensor<D, Chips, m![V]>> for ToDm<Hidden>
{
    type Output = Columns<Hidden, V, D, Chips>;

    fn forward(&self, device: &mut Device, vector: &HbmTensor<D, Chips, m![V]>) -> Self::Output {
        vector.view().to_dm(&mut device.tdma)
    }
}

impl<Hidden: AxisName, R: AxisName, Chips: M + 'static>
    Op<&gather::Rows<'_, bf16, Hidden, R, Chips>> for ToDm<Hidden>
{
    type Output = Spread<Hidden, R, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        input: &gather::Rows<'_, bf16, Hidden, R, Chips>,
    ) -> Spread<Hidden, R, Chips> {
        let (hidden, _) = input.stage(device);
        hidden.to_dm(&mut device.tdma)
    }
}

/// Runs `pass` on each pass's written weights, pass `p + 1`'s weights issued before pass `p` runs.
fn stream<Hidden: AxisName, V: AxisName, Chips: M + 'static>(
    device: &mut Device,
    weight: &HbmTensor<bf16, Chips, m![V, Hidden]>,
    mut pass: impl FnMut(&mut Device, &Weights<Chips, Hidden, V>, usize),
) {
    let mut current: Weights<Chips, Hidden, V> = weight
        .view()
        .tile::<m![V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })], 1, m![
            V / ({ Block<Hidden, V> }),
            V % ({ PassRows<Hidden, V> }) # ({ Block<Hidden, V> }), Hidden
        ]>(0)
        .to_dm(&mut device.tdma);
    #[unroll]
    for at in 0..<Passes<Hidden, V> as M>::SIZE {
        let mut next: Weights<Chips, Hidden, V> = DmTensor::new();
        if at + 1 < <Passes<Hidden, V> as M>::SIZE {
            weight
                .view()
                .tile::<m![V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })], 1, m![
                    V / ({ Block<Hidden, V> }),
                    V % ({ PassRows<Hidden, V> }) # ({ Block<Hidden, V> }), Hidden
                ]>(at + 1)
                .to_dm_view(&mut device.tdma, next.view_mut());
        }
        pass(&mut *device, &current, at);
        current = next;
    }
}

/// One pass's FP32 logits of `Count` lane groups of the rows, each plus its bias, finished by
/// `finish`.
fn biased<Hidden: AxisName, V: AxisName, Rows: AxisName, Count: M, T, Chips: M + 'static>(
    device: &mut Device,
    (weight, lanes, bias): (
        &Weights<Chips, Hidden, V>,
        &Lanes<Chips, Hidden, V, Rows, Count>,
        &Columns<Hidden, V, f32, Chips>,
    ),
    at: usize,
    finish: impl FnOnce(
        VectorFinalTensor<
            '_,
            { Tu::Main },
            f32,
            Chips,
            Cluster<V>,
            Reduced<Hidden, V>,
            m![V % ({ PassRows<Hidden, V> }), { Count }],
            m![{ Layout<Rows>::Lanes } # 8],
        >,
    ) -> T,
) -> T {
    let bias: VrfTensor<
        f32,
        Chips,
        Cluster<V>,
        Reduced<Hidden, V>,
        m![V % ({ PassRows<Hidden, V> })],
    > = device
        .sub
        .begin(
            bias.view()
                .tile::<
                    m![V % ({ Block<Hidden, V> }) / ({ PassRows<Hidden, V> })],
                    1,
                    Column<Hidden, V>,
                >(at),
        )
        .fetch::<m![V % ({ PassRows<Hidden, V> }) / 8], m![V % 8]>()
        .collect::<m![V % ({ PassRows<Hidden, V> }) / 8], m![V % 8]>()
        .to_vrf();
    let sums = kernel::Contraction::<Hidden, V>::sums(device, weight, lanes)
        .vector_init()
        .vector_inter_slice_reduce::<
            Reduced<Hidden, V>,
            m![V % ({ PassRows<Hidden, V> }), { Count }],
        >(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<
            m![V % ({ PassRows<Hidden, V> }), { Count }, { Layout<Rows>::Lanes } # 8 / 4],
            m![{ Layout<Rows>::Lanes } # 8 % 4],
        >()
        .vector_fp_binary(FpBinaryOp::AddF, &bias)
        .vector_widen_concat::<
            m![V % ({ PassRows<Hidden, V> }), { Count }],
            m![{ Layout<Rows>::Lanes } # 8],
        >()
        .vector_final();
    finish(sums)
}

/// `Count` lane groups of the input rows as the tensor-register lanes of every slice, each slice
/// the rows' features in its chunk.
fn stage<
    'i,
    Hidden: AxisName,
    V: AxisName,
    Rows: AxisName,
    Count: M,
    Element: M,
    Chips: M + 'static,
>(
    device: &mut Device,
    input: DmTensorView<'i, bf16, Chips, normalization::Cluster, FeatureSlices<Hidden>, Element>,
) -> Lanes<Chips, Hidden, V, Rows, Count> {
    type C<Hidden> = Chunk<Hidden>;
    type Slices<Hidden, V> = Slice<Hidden, V>;
    if const { Hidden::SIZE == 16 * 256 } {
        // Every slice holds sixteen features in both clusters, preserving the source addresses.
        let input = unsafe { input.reshape::<Chips, Cluster<V>, m![Hidden / 16], Element>() };
        device
            .sub
            .begin(input)
            .fetch::<m![{ Layout<Rows>::Lanes }, { Count }], m![Hidden % 16]>()
            .switch::<Slices<Hidden, V>, m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ]>(SwitchConfig::TransposedBroadcast1 {
                slice1: Hidden::SIZE / <C<Hidden> as M>::SIZE,
                slice0: <C<Hidden> as M>::SIZE / 16,
            })
            .collect::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ], m![Hidden % 16]>()
            .to_trf()
    } else if const { Hidden::SIZE > 16 * 256 } {
        // A slice holds several switch packets, delivered with other slices' parts innermost, so
        // the parts are written in DM before their lanes fill in feature order.
        let input = unsafe { input.reshape::<Chips, Cluster<V>, FeatureSlices<Hidden>, Element>() };
        let mut staged: DmTensor<
            bf16,
            Chips,
            Cluster<V>,
            Slices<Hidden, V>,
            m![{ Count }, { Layout<Rows>::Lanes }, Hidden % ({ C<Hidden> })],
        > = DmTensor::new();
        device
            .sub
            .begin(input)
            .fetch::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ Feature<Hidden> }) / 16
            ], m![Hidden % 16]>()
            .switch::<Slices<Hidden, V>, m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ Feature<Hidden> }) / 16,
                Hidden % ({ C<Hidden> }) / ({ Feature<Hidden> })
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ Feature<Hidden> }) / 16,
                Hidden % ({ C<Hidden> }) / ({ Feature<Hidden> })
            ], m![Hidden % 16]>()
            .commit_trim::<m![Hidden % 16]>()
            .commit_view(staged.view_mut());
        device
            .sub
            .begin(staged.view())
            .fetch::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ], m![Hidden % 16]>()
            .collect::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ], m![Hidden % 16]>()
            .to_trf()
    } else {
        // Both physical clusters already hold every row; only the logical cluster axis changes.
        let input = unsafe { input.reshape::<Chips, Cluster<V>, m![Hidden / 16 # 256], Element>() };
        device
            .sub
            .begin(input)
            .fetch::<m![{ Layout<Rows>::Lanes }, { Count }], m![Hidden % 16]>()
            .switch::<Slices<Hidden, V>, m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
            .collect::<m![
                { Layout<Rows>::Lanes },
                { Count },
                Hidden % ({ C<Hidden> }) / 16
            ], m![Hidden % 16]>()
            .to_trf()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{attention::geometry::Hidden, linear::Biased};
    use crate::tensor::axis::Rows;
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::launch;

    axes![Vocabulary = 8192];

    #[device(chip = 1)]
    fn scored<H: AxisName, V: AxisName, R: AxisName>(
        device: &mut Device,
        head: &Biased<H, V>,
        input: &HbmTensor<bf16, Chip, m![R, H]>,
    ) -> HbmTensor<f32, Chip, m![R, V]> {
        let input: Spread<H, R> = input.to_dm(&mut device.tdma);
        let bias: Columns<H, V, f32> = head.bias.view().to_dm(&mut device.tdma);
        let mut scores = HbmTensor::new();
        Affine::new(&head.weight, bias).forward(device, (&input, &mut scores));
        scores
    }

    /// Each row's scores are its features' products with each token's weights plus the token's
    /// bias, exactly, whether a slice holds sixteen features of a row or more.
    #[tokio::test(flavor = "current_thread")]
    async fn scores_each_row_at_any_hidden_width() {
        let _lease = DEVICE.lock().await;
        scores_each_row::<Hidden<1024>>().await;
        scores_each_row::<Hidden<5120>>().await;
    }

    /// Row `r` holds 1 at the last feature of slice `r` and 2 near the row's end; weights are
    /// sixteenths and biases carry bits BF16 drops, so every score is exact in FP32.
    async fn scores_each_row<H: AxisName>() {
        type R = Rows<16>;
        let slice = <Symbol<normalization::Chunk<H>> as M>::SIZE;
        let features = |row: usize| [row * slice + slice - 1, H::SIZE - 1 - row * (slice + 1)];
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let weight: Vec<bf16> = (0..Vocabulary::SIZE * H::SIZE)
            .map(|_| bf16::from_f32((next() >> 60) as f32 / 16.0 - 0.5))
            .collect();
        let bias: Vec<f32> = (0..Vocabulary::SIZE)
            .map(|_| (next() >> 54) as f32 / 1024.0 - 0.5)
            .collect();
        let mut input = vec![bf16::from_f32(0.0); R::SIZE * H::SIZE];
        for row in 0..R::SIZE {
            let [first, second] = features(row);
            input[row * H::SIZE + first] = bf16::from_f32(1.0);
            input[row * H::SIZE + second] = bf16::from_f32(2.0);
        }
        let mut device = Device::new(Topology { chips: 1, pes: 8 }).unwrap();
        let head = Biased::<H, Vocabulary>::new(
            HostTensor::<bf16, m![Vocabulary, H]>::from_vec(weight.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
            HostTensor::<f32, m![Vocabulary]>::from_vec(bias.clone())
                .to_hbm(&mut device.pdma)
                .await
                .unwrap(),
        );
        let input = HostTensor::<bf16, m![R, H]>::from_vec(input)
            .to_hbm::<Chip, m![R, H]>(&mut device.pdma)
            .await
            .unwrap();
        let scores = launch(scored, (&mut device, &head, &input))
            .await
            .unwrap()
            .to_host::<m![R, Vocabulary]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for row in 0..R::SIZE {
            let [first, second] = features(row);
            for token in 0..Vocabulary::SIZE {
                let coefficient = |feature: usize| weight[token * H::SIZE + feature].to_f32();
                assert_eq!(
                    scores[row * Vocabulary::SIZE + token],
                    coefficient(first) + 2.0 * coefficient(second) + bias[token],
                    "width {}, row {row}, token {token}",
                    H::SIZE,
                );
            }
        }
    }
}
