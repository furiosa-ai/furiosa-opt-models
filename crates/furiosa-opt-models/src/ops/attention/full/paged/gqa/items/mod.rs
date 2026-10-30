//! Decode attention over work items: an item is one row's keys on an aligned group of slices.
//!
//! - A launch's decode rows split by history length into two domains, and every launch runs
//!   both, a domain without rows no wave: rows of short histories pack two to a group of four-slice
//!   columns ([`packed`]), longer ones take groups of 16-slice columns ([`wide`]).
//! - Each domain runs one body over its waves; its wave count, span and limits are launch values.
//!   A row longer than a wide item takes consecutive items, which a merge combines after the waves.
//! - The packed domain runs first, and the caller's boundary work runs once, behind the launch's
//!   last history: in the wide domain's last wave, or the packed one's when no row is wide.
//! - A column holds at most four query groups of a KV head: a row of more groups attends them a
//!   [`GroupTile`] at a time, each tile an item row of its own over the row's history.

use super::*;
use crate::Module;
use crate::ops::attention;
use crate::tensor::memory::alias;
use crate::tensor::scalar::Value;
use furiosa_opt_std::runtime::HbmBuffer;

pub use plan::{Plan, Tiling};
use std::io;

mod attend;
mod packed;
mod plan;
mod wide;

/// The `N` waves a domain's items are compiled for.
#[derive(Clone, Debug)]
pub(crate) struct Waves<const N: usize>;

impl<const N: usize> AxisName for Waves<N> {
    const NAME: Ident = Ident::new("Waves");
    const SIZE: usize = N;
}

const CLEARED: TagGuard =
    TagGuard::matches([BitReq::Zero, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const ADMITTED: TagGuard =
    TagGuard::matches([BitReq::One, BitReq::Ignore, BitReq::Ignore, BitReq::Ignore]);
const SECOND_CLEARED: TagGuard =
    TagGuard::matches([BitReq::Ignore, BitReq::Zero, BitReq::Ignore, BitReq::Ignore]);
/// Rows past the attended rows' last ones take no writes: both domains' sinks lie in them.
pub(crate) const SINKS: usize = if packed::SINKS > wide::SINKS {
    packed::SINKS
} else {
    wide::SINKS
};

/// A launch's rows of every (KV head, group): its query rows, and the rows attention returns.
type HeadRows<R, KvHeads, G, Head> = HbmTensor<bf16, Chip, m![R, KvHeads, G, Head]>;

/// The query groups of one tile of `G`, at most four: an item attends a tile of each KV head.
#[derive(Clone, Debug)]
pub(crate) struct GroupTile<G>(PhantomData<G>);

impl<G: AxisName> GroupTile<G> {
    /// The tiles of `G`.
    const TILES: usize = G::SIZE.div_ceil(4);
}

impl<G: AxisName> AxisName for GroupTile<G> {
    const NAME: Ident = G::NAME;
    const SIZE: usize = G::SIZE / Self::TILES;
}

/// The rows of `R`, each as its tiles of `G`'s groups in turn: an item row a tile.
#[derive(Clone, Debug)]
pub(crate) struct Tiled<R, G>(PhantomData<(R, G)>);

impl<R: AxisName, G: AxisName> AxisName for Tiled<R, G> {
    const NAME: Ident = R::NAME;
    const SIZE: usize = R::SIZE * GroupTile::<G>::TILES;
}

/// A launch's decode rows as items of its two domains, planned on the host: the rows, and each
/// domain's launch counts and tables, `WAVES` wide waves. [`Plan`] makes the arrays.
#[derive(DeviceSend)]
pub struct Items<KvHeads: AxisName, G: AxisName, Head: AxisName, const WAVES: usize> {
    rows: Value,
    packed: packed::Packed<KvHeads, Head>,
    wide: wide::Wide<KvHeads, Head, Waves<WAVES>>,
    groups: PhantomData<G>,
}

/// The counts of an [`Items`], which a launch reads once, at its start.
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    packed: packed::Shape,
    wide: wide::Shape,
}

impl<KvHeads: AxisName, G: AxisName, Head: AxisName, const WAVES: usize>
    Items<KvHeads, G, Head, WAVES>
{
    /// The items whose arrays [`Plan::reads`] lists, as `next` names each in turn.
    pub fn bind(mut next: impl FnMut() -> io::Result<HbmBuffer>) -> io::Result<Self> {
        Ok(Self {
            rows: alias(&next()?, ..)?,
            packed: packed::Packed::bind(&mut next)?,
            wide: wide::Wide::bind(&mut next)?,
            groups: PhantomData,
        })
    }

    /// Bytes of each HBM allocation a layer's items make: every wide item's f32 output, and its
    /// log-sum-exp.
    pub(crate) const SCRATCH: [usize; 2] =
        wide::Wide::<KvHeads, Head, Waves<WAVES>>::scratch::<GroupTile<G>>();

    /// The rows the items attend, from row 0.
    pub(crate) fn rows(&self) -> &Value {
        &self.rows
    }

    /// The launch's counts.
    pub(crate) fn shape(&self) -> Shape {
        Shape {
            packed: self.packed.shape(),
            wide: self.wide.shape(),
        }
    }
}

/// The KV heads each of a chip's two clusters holds, at most four: half of a launch's heads, or its
/// one head on the first cluster.
#[derive(Clone, Debug)]
pub(crate) struct LocalHeads<KvHeads>(PhantomData<KvHeads>);

impl<KvHeads: AxisName> AxisName for LocalHeads<KvHeads> {
    const NAME: Ident = Ident::new("LocalHeads");
    const SIZE: usize = {
        let half = KvHeads::SIZE.div_ceil(2);
        if half < 4 { half } else { 4 }
    };
}

/// [`LocalHeads`] as a mapping a KV head's coordinates divide by.
type Local<KvHeads> = Symbol<LocalHeads<KvHeads>>;

impl<KvHeads: AxisName, Groups: AxisName, Head: AxisName> Attention<KvHeads, Groups, Head> {
    /// The cluster of each KV head, the first one only for a single head.
    type ItemCluster = m![KvHeads / ({ Local<KvHeads> }) # 2];
}

/// Attends each decode row of `items` over its history in `pool` into `attended`, a group tile `Q`
/// a query row; `boundary` runs once the last history is in flight.
impl<
    KvHeads: AxisName,
    Groups: AxisName,
    Head: AxisName,
    const WAVES: usize,
    Rows: AxisName,
    Q: AxisName,
    H,
    S,
    Env: Copy,
>
    Module<
        (
            &HeadRows<Rows, KvHeads, Q, Head>,
            (
                &Items<KvHeads, Groups, Head, WAVES>,
                Shape,
                (&Gather, &Blocks<KvHeads, Head>),
            ),
            &mut HeadRows<Rows, KvHeads, Q, Head>,
            (&H, &mut S),
        ),
        Env,
        &mut (),
    > for Attention<KvHeads, Groups, Head>
where
    H: for<'d, 's> Module<&'s mut S, Env, &'d mut (), Output = ()>,
{
    type Output = ();

    fn forward(
        &self,
        context: &mut Device,
        env: Env,
        _: &mut (),
        (query, (items, shape, pool), attended, (boundary, state)): (
            &HeadRows<Rows, KvHeads, Q, Head>,
            (
                &Items<KvHeads, Groups, Head, WAVES>,
                Shape,
                (&Gather, &Blocks<KvHeads, Head>),
            ),
            &mut HeadRows<Rows, KvHeads, Q, Head>,
            (&H, &mut S),
        ),
    ) {
        const {
            assert!(
                Groups::SIZE <= 4 || Groups::SIZE.is_multiple_of(4),
                "a row's query groups split into tiles of four"
            );
            assert!(
                Q::SIZE == GroupTile::<Groups>::SIZE,
                "an item row holds one group tile of every KV head"
            );
            assert!(
                KvHeads::SIZE >= 4 && KvHeads::SIZE <= 8,
                "a launch's KV heads fill the two clusters, two to four a cluster"
            );
        };
        let tile: Attention<KvHeads, Q, Head> = attention::Attention {
            heads: PhantomData,
            scale: self.scale,
        };
        let alone = 1 - shape.wide.waves.min(1);
        let packed = (&items.packed, shape.packed, pool);
        tile.forward(
            &mut *context,
            env,
            &mut (),
            (
                (query, &mut *attended),
                packed,
                ((boundary, &mut *state), alone),
            ),
        );
        let wide = (&items.wide, shape.wide, pool);
        tile.forward(
            context,
            env,
            &mut (),
            ((query, attended), wide, ((boundary, state), 1)),
        );
    }
}

#[cfg(test)]
pub(crate) mod tests {

    pub(crate) use self::{
        Eights as QueryEights, Groups as QueryGroups, Head as Width, KvHeads as Heads,
        Pairs as QueryPairs, Single as QuerySingle, Sixteens as QuerySixteens,
    };
    use super::*;
    use crate::specialize::attention::{DECODES, HISTORY_WAVES, WAVES};
    use crate::tensor::axis::Rows;
    use crate::tensor::cache::{Block, Blocks};
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::tests::DEVICE;
    use furiosa_opt_std::runtime::{HostBufferSlice, launch};
    use std::f32::consts::FRAC_1_SQRT_2;
    use std::vec::IntoIter;

    axes![
        KvHeads = 8,
        FourHeads = 4,
        Groups = 4,
        Pairs = 2,
        Single = 1,
        Eights = 8,
        Sixteens = 16,
        Head = 128,
        Layers = 16
    ];

    /// The attended rows: the decode rows, rows past them, and the columns' sinks last.
    type Reach = Rows<1280>;
    type Layout<H, G> = m![{ Symbol<Tiled<Reach, G>> }, H, { Symbol<GroupTile<G>> }, Head];
    type Queries<H, G> = HbmTensor<bf16, Chip, Layout<H, G>>;
    pub(crate) type Pool<H = KvHeads> = Blocks<H, Head>;

    /// The score multiplier of 128-wide heads, `1 / sqrt(128)`.
    pub(crate) const SCALE: f32 = FRAC_1_SQRT_2 / 8.0;
    /// Keys of each of 32 short histories, 1,088 cached and the current one, and of the long one
    /// that rows past them read.
    pub(crate) const KEYS: usize = 1089;
    pub(crate) const LONG: usize = 40960;

    /// Histories of many lengths, 1 to 1,089 keys, block and item edges among them.
    pub(crate) const MIXED: [usize; 16] = [
        1, 17, 65, 257, 1089, 16, 255, 1024, 64, 2, 1088, 31, 512, 129, 700, 48,
    ];

    const UNWRITTEN: f32 = 7.0;

    /// Channels of a row of every (KV head, group) of `H` heads of `G` groups.
    pub(crate) const fn stride<H: AxisName, G: AxisName>() -> usize {
        <m![H, G, Head]>::SIZE
    }

    #[device(chip = 1)]
    fn attend_items<H: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        query: &Queries<H, G>,
        items: &Items<H, G, Head, WAVES>,
        layers: &Value,
        pool: &mut Pool<H>,
        attended: &mut Queries<H, G>,
    ) {
        let attention: Attention<H, G, Head> = attention::Attention {
            heads: PhantomData,
            scale: SCALE,
        };
        let shape = items.shape();
        for _ in 0..layers.extent::<m![Layers]>() {
            let input = (items, shape, (&Gather, &*pool));
            Module::forward(
                &attention,
                &mut *device,
                (),
                &mut (),
                (query, input, &mut *attended, (&(), &mut ())),
            );
        }
    }

    /// Row `row`'s query of KV head `head` and group `group` on every channel, exact in BF16; each
    /// tile of four groups differs from the others.
    fn query(row: usize, (head, group): (usize, usize)) -> f32 {
        let row = (row % 8) as f32 / 8.0 + (row / 32 % 4) as f32 / 32.0;
        let group = (group % 4) as f32 / 4.0 + (group / 4) as f32 / 8.0;
        0.5 + row + group - head as f32 / 8.0
    }

    /// `rows` query rows of `H` KV heads of `G` groups each.
    pub(crate) fn queries<H: AxisName, G: AxisName>(rows: usize) -> impl Iterator<Item = bf16> {
        (0..rows * stride::<H, G>()).map(|at| {
            let (row, at) = (at / stride::<H, G>(), at % stride::<H, G>() / Head::SIZE);
            bf16::from_f32(query(row, (at / G::SIZE, at % G::SIZE)))
        })
    }

    /// Where channel `at` of rows of `H` KV heads of `G` groups lies once each row is its group
    /// tiles: tile `t` of row `r` is item row `r T + t`, groups `4t..4t + 4` of every head.
    fn tiled<H: AxisName, G: AxisName>(at: usize) -> usize {
        let tile = GroupTile::<G>::SIZE;
        let (row, head) = (at / stride::<H, G>(), at / (G::SIZE * Head::SIZE) % H::SIZE);
        let (group, channel) = (at / Head::SIZE % G::SIZE, at % Head::SIZE);
        let item = row * GroupTile::<G>::TILES + group / tile;
        ((item * H::SIZE + head) * tile + group % tile) * Head::SIZE + channel
    }

    /// Token `token`'s key or value of KV head `head` in history `history`; channel `c` of a value
    /// holds it divided by `2^(c % 4)`, and every value is exact in BF16.
    fn key_value(history: usize, token: usize, (kind, head): (usize, usize)) -> f32 {
        match kind {
            0 => ((token * 3 + head + history) % 17) as f32 / 32.0 - 0.25,
            _ => (((token * 3 + head * 5 + history * 7) % 29) as f32 - 14.0) / 16.0,
        }
    }

    /// Each of `reads`' arrays on `device` in a buffer of its own that holds only the words the
    /// launch uses.
    pub(crate) async fn place<'a>(
        device: &mut Device,
        reads: impl IntoIterator<Item = (&'a [i32], usize)>,
    ) -> IntoIter<HbmBuffer> {
        let mut buffers = Vec::new();
        for (read, words) in reads {
            let mut buffer = HbmBuffer::alloc(device, words * size_of::<i32>()).unwrap();
            let bytes: Vec<u8> = read.iter().flat_map(|word| word.to_le_bytes()).collect();
            let target = buffer.slice_mut(.., ..bytes.len()).unwrap();
            let transfer = [(HostBufferSlice::from(&bytes[..]), target)];
            device.pdma.to_hbm(transfer).await.unwrap();
            buffers.push(buffer);
        }
        buffers.into_iter()
    }

    /// The keys of history `history`: 32 short ones, then the long one.
    pub(crate) fn keys(history: usize) -> usize {
        if history < 32 { KEYS } else { LONG }
    }

    /// The history row `row` reads when it sees `length` keys: the long one past `KEYS`.
    pub(crate) fn history(row: usize, length: usize) -> usize {
        if length > KEYS { 32 } else { row % 32 }
    }

    /// The f64 softmax of row `row` over its `length` keys, for KV head `head` and group `group`.
    fn expected(row: usize, length: usize, (head, group): (usize, usize)) -> f64 {
        let history = history(row, length);
        let score = |token| {
            let key = f64::from(key_value(history, token, (0, head)));
            f64::from(query(row, (head, group))) * key * 128.0 * f64::from(SCALE)
        };
        let top = (0..length).map(score).fold(f64::NEG_INFINITY, f64::max);
        let (total, mass) = (0..length).fold((0.0, 0.0), |(total, mass), token| {
            let weight = (score(token) - top).exp();
            (
                total + weight * f64::from(key_value(history, token, (1, head))),
                mass + weight,
            )
        });
        total / mass
    }

    /// The pool of 33 histories of `H` KV heads: 32 short ones, then the long one, each one's
    /// blocks in reverse order and holding NaN or infinities past its keys; and each history's
    /// first block.
    pub(crate) async fn pool<H: AxisName>(device: &mut Device) -> (Pool<H>, Vec<usize>) {
        let blocks = |history: usize| keys(history).div_ceil(Block::SIZE);
        let firsts: Vec<usize> = (0..33)
            .map(|history| (0..history).map(blocks).sum())
            .collect();
        let width = KeyValue::SIZE * H::SIZE * Head::SIZE;
        let rows = (0..33).flat_map(|history| {
            (0..blocks(history)).rev().flat_map(move |block| {
                (0..Block::SIZE * width).map(move |at| {
                    let (token, kind) = (block * Block::SIZE + at / width, at / (width / 2) % 2);
                    let (head, channel) = (at / Head::SIZE % H::SIZE, at % Head::SIZE);
                    let poison = [f32::NAN, -f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
                    bf16::from_f32(match token < keys(history) {
                        true => {
                            key_value(history, token, (kind, head))
                                / (1 << (kind * (channel % 4))) as f32
                        }
                        false => poison[(token + channel + kind) % 4],
                    })
                })
            })
        });
        (Pool::<H>::from_rows(device, rows.collect()).await, firsts)
    }

    /// The items within `WAVES` wide waves of rows of `H` KV heads of `G` groups of `firsts`' pool,
    /// in `R`'s rows, tiled as `tiling` says: row `r` sees a prefix of `rows[r].1` keys of fixture
    /// row `rows[r].0`'s history, their block IDs padded to 16 with its last.
    pub(crate) fn plan<H: AxisName, G: AxisName, const WAVES: usize, R: AxisName>(
        rows: impl IntoIterator<Item = (usize, usize)>,
        firsts: &[usize],
        tiling: Tiling,
    ) -> Plan {
        let blocks = |history: usize| keys(history).div_ceil(Block::SIZE);
        let (mut segments, mut ids) = (Vec::new(), Vec::new());
        for (row, (source, length)) in rows.into_iter().enumerate() {
            let (history, used) = (history(source, length), length.div_ceil(Block::SIZE));
            segments.push([row, 1, length - 1, ids.len(), used].map(|field| field as i32));
            let last = firsts[history] + blocks(history) - 1;
            ids.extend((0..used.next_multiple_of(16)).map(|b| (last - b.min(used - 1)) as i32));
        }
        Plan::new::<H, G, Head, WAVES, R>(&segments, &ids, tiling)
    }

    /// The largest distance of row `row`'s attention over its `length` keys for `H` KV heads of `G`
    /// groups from the f64 softmax.
    pub(crate) fn error<H: AxisName, G: AxisName>(
        row: usize,
        length: usize,
        output: &[bf16],
    ) -> f64 {
        let means: Vec<f64> = (0..H::SIZE * G::SIZE)
            .map(|at| expected(row, length, (at / G::SIZE, at % G::SIZE)))
            .collect();
        let distance = |(at, found): (usize, &bf16)| {
            let want = means[at / Head::SIZE] / f64::from(1u32 << (at % Head::SIZE % 4));
            (f64::from(found.to_f32()) - want).abs()
        };
        output.iter().enumerate().map(distance).fold(0.0, f64::max)
    }

    /// Row `row`'s attention over its `length` keys for `H` KV heads of `G` groups, as a host
    /// softmax gives it.
    pub(crate) fn check<H: AxisName, G: AxisName>(
        row: usize,
        length: usize,
        output: &[bf16],
        live: usize,
    ) {
        let means: Vec<f64> = (0..H::SIZE * G::SIZE)
            .map(|at| expected(row, length, (at / G::SIZE, at % G::SIZE)))
            .collect();
        for (at, found) in output.iter().enumerate() {
            let (head, group) = (at / Head::SIZE / G::SIZE, at / Head::SIZE % G::SIZE);
            let channel = at % Head::SIZE;
            let want = means[at / Head::SIZE] / f64::from(1u32 << (channel % 4));
            let found = found.to_f32();
            assert!(
                (f64::from(found) - want).abs() <= 1.0 / 128.0,
                "{live} live: row {row} ({length} keys) head {head} group {group} channel \
                 {channel}: {found} != {want}",
            );
        }
    }

    /// Any number of live rows of eight or four KV heads and one to sixteen query groups a head,
    /// of one history length or many, short and long together, in any order, short of one item or
    /// many items long, each attend their own prefix of a paged history as a host softmax does
    /// under either tiling, though the cache holds NaN past it and its blocks lie in reverse
    /// order, twice over one derivation; rows past the live ones keep their bytes. Within
    /// [`HISTORY_WAVES`], 64 rows of the whole history fill every wide wave, an odd count of them
    /// leaves a last wave alone, and a long row beside short ones and every other row long take
    /// their own items.
    #[tokio::test(flavor = "current_thread")]
    async fn attends_each_live_row_once_derived() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_items.topology()).unwrap();
        let (mut eight, firsts) = pool::<KvHeads>(&mut device).await;
        let (mut four, _) = pool::<FourHeads>(&mut device).await;
        let cases = cases();
        let device = &mut device;
        let whole = [
            vec![LONG; 64],
            vec![LONG; 63],
            [LONG].into_iter().chain([25; 63]).collect(),
            (0..64).map(|row| [LONG, 1][row % 2]).collect(),
        ];
        for tiling in [Tiling::Adaptive, Tiling::Fixed] {
            let cases = (&cases[..], tiling);
            attends::<KvHeads, Groups, WAVES>(device, (&mut eight, &firsts), cases).await;
            attends::<KvHeads, Pairs, WAVES>(device, (&mut eight, &firsts), cases).await;
            attends::<KvHeads, Single, WAVES>(device, (&mut eight, &firsts), cases).await;
            attends::<KvHeads, Eights, WAVES>(device, (&mut eight, &firsts), cases).await;
            attends::<FourHeads, Groups, WAVES>(device, (&mut four, &firsts), cases).await;
            attends::<FourHeads, Eights, WAVES>(device, (&mut four, &firsts), cases).await;
            attends::<FourHeads, Sixteens, WAVES>(device, (&mut four, &firsts), cases).await;
            let whole = (&whole[..], tiling);
            attends::<KvHeads, Groups, HISTORY_WAVES>(device, (&mut eight, &firsts), whole).await;
        }
    }

    /// Under a fixed tiling a row attends to the same bits whatever rows share its launch and
    /// wherever it lies in it: row 0's query over prefixes of a short and of the long history on
    /// each side of the packed bound and of an item's edge, alone and at the first, a middle and
    /// the last row of 16 beside short and long rows and of 64 of the whole history; alone, it
    /// attends as a host softmax does.
    #[tokio::test(flavor = "current_thread")]
    async fn keeps_row_bits_under_fixed_tiling() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_items.topology()).unwrap();
        let (mut pool, firsts) = pool::<KvHeads>(&mut device).await;
        let device = &mut device;
        let beside = [25, LONG, 1, 4097, 2048, 300, 17, 16385];
        let stride = stride::<KvHeads, Groups>();
        let fixture: Vec<bf16> = queries::<KvHeads, Groups>(Reach::SIZE).collect();
        // Each launch row's query is the one of the fixture row whose history it sees.
        let carried = |rows: &[(usize, usize)]| {
            let mut queries = fixture.clone();
            for (row, &(source, _)) in rows.iter().enumerate() {
                queries[row * stride..][..stride]
                    .copy_from_slice(&fixture[source * stride..][..stride]);
            }
            queries
        };
        let bits = |output: &[bf16], row: usize| {
            let values = &output[row * stride..][..stride];
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        for length in [
            1, 17, 255, 256, 257, KEYS, 4095, 4096, 4097, 8193, 20481, LONG,
        ] {
            let rows = [(0, length)];
            let histories = (&mut pool, &firsts[..]);
            let fixed = (&rows[..], Tiling::Fixed);
            let alone =
                attend::<KvHeads, Groups, WAVES>(device, histories, fixed, carried(&rows)).await;
            check::<KvHeads, Groups>(0, length, &alone[..stride], 1);
            for at in [0, 7, 15] {
                let rows: Vec<_> = (0..16)
                    .map(|row| match row == at {
                        true => (0, length),
                        false => (row, beside[row % beside.len()]),
                    })
                    .collect();
                let (histories, fixed) = ((&mut pool, &firsts[..]), (&rows[..], Tiling::Fixed));
                let shared =
                    attend::<KvHeads, Groups, WAVES>(device, histories, fixed, carried(&rows))
                        .await;
                let row = format!("{length} keys at row {at} of 16");
                assert_eq!(bits(&shared, at), bits(&alone, 0), "{row}");
            }
            for at in [0, 33, 63] {
                let rows: Vec<_> = (0..64)
                    .map(|row| match row == at {
                        true => (0, length),
                        false => (row, LONG),
                    })
                    .collect();
                let (histories, fixed) = ((&mut pool, &firsts[..]), (&rows[..], Tiling::Fixed));
                let shared = attend::<KvHeads, Groups, HISTORY_WAVES>(
                    device,
                    histories,
                    fixed,
                    carried(&rows),
                )
                .await;
                let row = format!("{length} keys at row {at} of 64");
                assert_eq!(bits(&shared, at), bits(&alone, 0), "{row}");
            }
        }
    }

    /// Rows of one history length or many, short and long together, in any order, short of one
    /// item or many items long.
    fn cases() -> Vec<Vec<usize>> {
        // Short rows beside long ones, and beside a row of many items.
        let short = [25; 512].into_iter();
        let both = [
            short.clone().chain([4096; 32]).collect(),
            short.chain([LONG; 2]).collect(),
        ];
        let many = [LONG].into_iter().chain([17; 199]).collect();
        let cases = [0, 1, 8, 16, 30, 32, 33, 64].map(|live| vec![KEYS; live]);
        let paired = [1, 65, 129, 1024].map(|live| vec![25; live]);
        cases
            .into_iter()
            .chain(paired)
            .chain(both)
            .chain([
                (0..64).map(|row| MIXED[row % 16]).collect(),
                (0..64).map(|row| MIXED[15 - row % 16]).collect(),
                vec![16384],
                many,
                vec![16385, 1, 20000, 1089, LONG, 17, 2, 16384],
            ])
            .collect()
    }

    /// Every case of `cases` whose rows' group tiles a launch holds, rows of `H` KV heads of `G`
    /// groups tiled as `tiling` says: each live row attends as a host softmax does, and rows past
    /// them keep their bytes.
    async fn attends<H: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        (pool, firsts): (&mut Pool<H>, &[usize]),
        (cases, tiling): (&[Vec<usize>], Tiling),
    ) {
        let unwritten = bf16::from_f32(UNWRITTEN);
        let held = |lengths: &&Vec<usize>| lengths.len() * GroupTile::<G>::TILES <= DECODES;
        for lengths in cases.iter().filter(held) {
            let rows: Vec<_> = lengths.iter().copied().enumerate().collect();
            let fixture = queries::<H, G>(Reach::SIZE).collect();
            let histories = (&mut *pool, firsts);
            let output = attend::<H, G, WAVES>(device, histories, (&rows, tiling), fixture).await;
            for (row, got) in output
                .chunks_exact(stride::<H, G>())
                .enumerate()
                .take(Reach::SIZE - SINKS)
            {
                match lengths.get(row) {
                    Some(&length) => check::<H, G>(row, length, got, lengths.len()),
                    None => assert!(
                        got.iter().all(|value| *value == unwritten),
                        "row {row} written"
                    ),
                }
            }
        }
    }

    /// A launch's attended rows, twice over, tiled as `tiling` says: row `r` sees `rows[r].1` keys
    /// of fixture row `rows[r].0`'s history in `firsts`' pool; `queries` holds every row of `Reach`.
    async fn attend<H: AxisName, G: AxisName, const WAVES: usize>(
        device: &mut Device,
        (pool, firsts): (&mut Pool<H>, &[usize]),
        (rows, tiling): (&[(usize, usize)], Tiling),
        queries: Vec<bf16>,
    ) -> Vec<bf16> {
        let mut tiles = queries.clone();
        for (at, value) in queries.into_iter().enumerate() {
            tiles[tiled::<H, G>(at)] = value;
        }
        let queries: Queries<H, G> = HostTensor::<bf16, Layout<H, G>>::from_vec(tiles)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let layers = count(2).to_hbm(&mut device.pdma).await.unwrap();
        let plan = plan::<H, G, WAVES, Reach>(rows.iter().copied(), firsts, tiling);
        let mut buffers = place(device, plan.reads()).await;
        let items: Items<H, G, Head, WAVES> =
            Items::bind(|| Ok(buffers.next().expect("an array a buffer"))).unwrap();
        let unwritten_rows = vec![bf16::from_f32(UNWRITTEN); Reach::SIZE * stride::<H, G>()];
        let mut attended: Queries<H, G> =
            HostTensor::<bf16, Layout<H, G>>::from_vec(unwritten_rows)
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let arguments = (
            &mut *device,
            &queries,
            &items,
            &layers,
            &mut *pool,
            &mut attended,
        );
        launch(attend_items, arguments).await.unwrap();
        let tiles = attended
            .to_host::<Layout<H, G>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        (0..tiles.len())
            .map(|at| tiles[tiled::<H, G>(at)])
            .collect()
    }

    /// Query rows of NaN or infinities past the live rows give the finite outputs zeroed rows give.
    #[tokio::test(flavor = "current_thread")]
    async fn ignores_query_rows_past_the_live_ones() {
        let _lease = DEVICE.lock().await;
        let mut device = Device::new(attend_items.topology()).unwrap();
        let (mut pool, firsts) = pool::<KvHeads>(&mut device).await;
        let cases: [Vec<usize>; 4] = [
            vec![KEYS; 1],
            vec![25; 65],
            (0..33).map(|row| MIXED[row % 16]).collect(),
            vec![16385, 1, LONG],
        ];
        for lengths in cases {
            let live = lengths.len() * stride::<KvHeads, Groups>();
            let seen: Vec<_> = lengths.iter().copied().enumerate().collect();
            let adaptive = (&seen[..], Tiling::Adaptive);
            let rows = |past: f32| {
                let query = queries::<KvHeads, Groups>(Reach::SIZE).enumerate();
                let fill = |(at, query): (usize, bf16)| match at < live {
                    true => query,
                    false => bf16::from_f32(past),
                };
                query.map(fill).collect()
            };
            let history = (&mut pool, &firsts[..]);
            let zeroed =
                attend::<KvHeads, Groups, WAVES>(&mut device, history, adaptive, rows(0.0)).await;
            for past in [f32::NAN, f32::INFINITY] {
                let history = (&mut pool, &firsts[..]);
                let got =
                    attend::<KvHeads, Groups, WAVES>(&mut device, history, adaptive, rows(past))
                        .await;
                for (at, (got, zeroed)) in got[..live].iter().zip(&zeroed[..live]).enumerate() {
                    let (got, zeroed) = (got.to_f32(), zeroed.to_f32());
                    assert!(
                        got.is_finite() && got == zeroed,
                        "{lengths:?} past={past} row={}: {got} != {zeroed}",
                        at / stride::<KvHeads, Groups>()
                    );
                }
            }
        }
    }
}
