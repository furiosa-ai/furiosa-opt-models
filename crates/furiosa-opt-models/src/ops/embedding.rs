//! Token features gathered from an embedding table: the table row of each token id.

use crate::load::{Checkpoint, Context, Load, LoadError};
use crate::modules::generic::schedule::{Fill, IntoStream, Stream};
use crate::ops::index::gather::Gather;
use crate::ops::index::table::{ByteOffset, Offset};
use crate::ops::linear::Groups;
use crate::ops::normalization::span::{self, Owned};
use crate::ops::normalization::{Blocked, Chunk, Spread};
use crate::ops::pointwise::{Float, add::Add};
use crate::specialize::index::{
    FEATURES,
    embedding::{BLOCK, REST, SPAN},
};
use crate::tensor::form::Weight;
use crate::tensor::mapping::{self, Clusters, Packets, Rows as Layout};
use crate::tensor::parallel::Chip;
use crate::tensor::{
    axis::{self, Lanes},
    live::Live,
};
use crate::{Module, Op};
use core::task::Poll;
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::HostBuffer;
use std::borrow::Borrow;
use std::marker::PhantomData;

/// The ids of `Rows` tokens, a packet of eight per row group as the DMA moves them; the lanes past
/// `Rows` are ignored.
pub type Ids<Rows, Chips = Chip> = HbmTensor<i32, Chips, Packets<Rows>>;

/// Look up token features by a row byte offset or by token IDs.
#[derive(DeviceSend, Load)]
#[load(tree)]
pub struct Embedding<
    Vocabulary: AxisName,
    Hidden: AxisName,
    Chips: M = m![1],
    Parameter = Weight<m![Vocabulary, Hidden], Chips>,
> {
    weight: Parameter,
    features: PhantomData<(Vocabulary, Hidden, Chips)>,
}

impl<Vocabulary: AxisName, Hidden: AxisName, Chips: M, Parameter>
    Embedding<Vocabulary, Hidden, Chips, Parameter>
{
    pub const fn new(weight: Parameter) -> Self {
        Self {
            weight,
            features: PhantomData,
        }
    }
}

/// A token table checked without its values, which reads them when a leaf calls it.
pub type Table<'a, Vocabulary, Hidden, Chips> =
    dyn Fn() -> Result<HostTensor<bf16, m![{ Chips }, { m![Vocabulary, Hidden] }]>, LoadError> + 'a;

impl<Vocabulary: AxisName, Hidden: AxisName, Chips: M> Embedding<Vocabulary, Hidden, Chips> {
    /// The config's `vocab_size`: the tokens a table holds, at most `Vocabulary`.
    pub fn tokens(source: &Checkpoint, context: &mut Context<'_>) -> Result<usize, LoadError> {
        let (name, tokens) = context.config_value(source, "vocab_size")?;
        tokens
            .as_u64()
            .and_then(|count| usize::try_from(count).ok())
            .filter(|count| (1..=Vocabulary::SIZE).contains(count))
            .ok_or_else(|| LoadError::InvalidData {
                name,
                reason: format!(
                    "{tokens} is not a token count of at most {}",
                    Vocabulary::SIZE
                ),
            })
    }

    /// The token table at the context's name, checked without its values: `vocab_size` BF16 or FP32
    /// rows of `Hidden`, at most `Vocabulary`; a call reads them as BF16, zero-filled past them.
    pub fn table<'a>(
        source: &'a Checkpoint,
        context: &mut Context<'_>,
    ) -> Result<Box<Table<'a, Vocabulary, Hidden, Chips>>, LoadError> {
        let rows = Self::tokens(source, context)?;
        let (name, (dtype, shape)) = context.header(source)?;
        Checkpoint::admit_bf16(dtype, &name)?;
        if shape != [rows, Hidden::SIZE] {
            return Err(LoadError::InvalidData {
                name,
                reason: format!(
                    "shape {shape:?}, the leaf holds the config's {rows} rows of {}",
                    Hidden::SIZE
                ),
            });
        }
        Ok(Box::new(move || {
            let view = source.get(&name).expect("declared above");
            let mut table = Checkpoint::bf16(&view, &name)?.into_owned();
            table.resize(Vocabulary::SIZE * Hidden::SIZE * size_of::<bf16>(), 0);
            let table = HostTensor::try_from(HostBuffer::from_vec(table.repeat(Chips::SIZE)));
            table.map_err(|error| LoadError::InvalidData {
                name: name.clone(),
                reason: error.to_string(),
            })
        }))
    }
}

/// The table under segment `weight`, checked as [`Embedding::table`] checks it; a leaf that holds
/// values reads them.
impl<'a, Vocabulary: AxisName, Hidden: AxisName, Chips: M, Parameter> Load<&'a Checkpoint>
    for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: for<'t> Load<&'t Table<'t, Vocabulary, Hidden, Chips>>,
{
    fn load(context: &mut Context<'_>, from: &'a Checkpoint) -> Result<Self, LoadError> {
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.compare(from)?;
                let table = Embedding::<Vocabulary, Hidden, Chips>::table(from, cx)?;
                Ok(Self::new(Load::load(cx, &*table)?))
            })
        })
    }
}

/// The table written in place: one rule of an update. By hand, with the build above: a derived
/// `with` update copies a new leaf through host.
impl<'t, 's, Vocabulary: AxisName, Hidden: AxisName, Chips: M, Parameter: 'static>
    Load<(
        &'t mut Embedding<Vocabulary, Hidden, Chips, Parameter>,
        &'s Checkpoint,
    )> for &'t mut Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    for<'u> &'u mut Parameter: Load<(
        &'u mut Parameter,
        HostTensor<bf16, m![{ Chips }, { m![Vocabulary, Hidden] }]>,
    )>,
{
    fn load(
        context: &mut Context<'_>,
        (into, from): (
            &'t mut Embedding<Vocabulary, Hidden, Chips, Parameter>,
            &'s Checkpoint,
        ),
    ) -> Result<Self, LoadError> {
        let write = |cx: &mut Context<'_>| {
            let table = Embedding::<Vocabulary, Hidden, Chips>::table(from, cx)?;
            <&mut Parameter as Load<_>>::load(cx, (&mut into.weight, table()?)).map(drop)
        };
        context.checkpoint(from, |cx| {
            cx.field("weight", "weight", |cx| {
                cx.rule(|cx| cx.overwrite::<Parameter>(from, write), |_, ()| Ok(()))
            })
        })?;
        Ok(into)
    }
}

// Keep row axes visible: associated mappings hide Rows from inference between composed modules.
pub type Vectors<Hidden, Rows, Chips = Chip> = DmTensor<
    bf16,
    Chips,
    m![Rows / ({ Symbol<Lanes<Rows>> }) % ({ Symbol<Clusters<Rows>> }) # 2],
    m![Rows % ({ Symbol<Lanes<Rows>> }) # 8, Hidden % { FEATURES } / 16],
    m![
        Hidden / { FEATURES },
        Rows / ({ Symbol<Lanes<Rows>> }) / ({ Symbol<Clusters<Rows>> }),
        Hidden % 16
    ],
>;

impl<Vocabulary: AxisName, Hidden: AxisName, Parameter, Chips: M + 'static>
    Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    /// The rows at byte `offsets`, a gather a feature tile: each gather reads its tile's view of
    /// the table at the rows' own offsets.
    fn tiles<Rows: AxisName, O: M>(
        &self,
        device: &mut Device,
        offsets: &HbmTensor<i32, Chips, O>,
    ) -> Vectors<Hidden, Rows, Chips> {
        let weight: &HbmTensor<bf16, Chips, m![Vocabulary, Hidden]> = self.weight.borrow();
        let mut output = DmTensor::new();
        for tile in 0..(Hidden::SIZE / FEATURES) {
            let gathered: DmTensor<
                bf16,
                Chips,
                m![{ Layout<Rows>::Spatial } # 2],
                m![{ Layout<Rows>::Lanes } # 8, Hidden % { FEATURES } / 16],
                m![{ Layout<Rows>::Serial }, Hidden % 16],
            > = weight
                .view()
                .tile::<m![Hidden / { FEATURES }], 1, m![
                    Vocabulary, Hidden / { FEATURES } = 1 # (Hidden / { FEATURES }),
                    Hidden % { FEATURES }
                ]>(tile)
                .gather::<m![Vocabulary], m![
                    Hidden / { FEATURES } = 1 # (Hidden / { FEATURES }), Hidden % { FEATURES }
                ]>()
                .by_byte_offsets(offsets)
                .to_dm(&mut device.tdma);
            gathered.view().to_dm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Hidden / { FEATURES }], 1, m![
                    1 #{!} (Hidden / { FEATURES }), { Layout<Rows>::Serial }, Hidden % 16
                ]>(tile),
            );
        }
        output
    }
}

/// Select one embedding row by its byte offset.
impl<Vocabulary: AxisName, Hidden: AxisName, Chips: M, Parameter> Op<&Offset<Chips>>
    for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, device: &mut Device, offset: &Offset<Chips>) -> Self::Output {
        let weight: &HbmTensor<bf16, Chips, m![Vocabulary, Hidden]> = self.weight.borrow();
        let row: DmTensor<bf16, Chips, m![1 # 2], m![1 # 256], m![Hidden]> = weight
            .gather::<m![Vocabulary], m![Hidden]>()
            .by_byte_offsets(offset)
            .to_dm(&mut device.tdma);
        row.to_hbm(&mut device.tdma)
    }
}

/// Select the row of one token ID, at the byte offset of its row.
impl<Vocabulary: AxisName, Hidden: AxisName, Chips: M, Parameter> Op<&HbmTensor<i32, Chips, m![1]>>
    for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    type Output = HbmTensor<bf16, Chips, m![Hidden]>;

    fn forward(&self, device: &mut Device, id: &HbmTensor<i32, Chips, m![1]>) -> Self::Output {
        let offset = Op::forward(&ByteOffset::<bf16, m![Hidden]>::default(), &mut *device, id);
        Op::forward(self, device, &offset)
    }
}

impl<'x, Vocabulary: AxisName, Hidden: AxisName, Rows: AxisName, Parameter, Chips: M + 'static>
    Op<&'x HbmTensor<i32, Chips, Packets<Rows>>> for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    type Output = Vectors<Hidden, Rows, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        ids: &'x HbmTensor<i32, Chips, Packets<Rows>>,
    ) -> Self::Output {
        let offsets = Op::forward(
            &ByteOffset::<bf16, m![Hidden], m![Rows]>::default(),
            &mut *device,
            ids,
        );
        self.tiles(device, &offsets)
    }
}

/// The rows of a launch's tokens in the decoder's feature mapping, with the launch's live rows.
impl<
    'x,
    Vocabulary: AxisName,
    Hidden: AxisName,
    Rows: AxisName,
    Parameter,
    Env: Copy,
    Chips: M + 'static,
> Module<(&'x Ids<Rows, Chips>, Live<(), Rows>), Env, &mut ()>
    for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    type Output = Live<Spread<Hidden, Rows, Chips>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        (ids, tiles): (&'x Ids<Rows, Chips>, Live<(), Rows>),
    ) -> Self::Output {
        let rows = tiles.with(Module::forward(&self, &mut *device, env, &mut (), ids));
        Module::forward(&Reorder::<Hidden>::default(), device, env, &mut (), rows)
    }
}

/// A launch's tokens, its live rows and the rows their features are written in.
pub(crate) struct Launch<'a, Hidden: AxisName, Capacity: AxisName, Chips: M + 'static = Chip> {
    pub(crate) ids: &'a Ids<Capacity, Chips>,
    pub(crate) count: Live<(), Capacity>,
    pub(crate) output: &'a mut HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>,
}

/// The live rows' features: whole spans of whole rows, then blocks of feature tiles, then a last
/// short gather, at the lengths of the specialized embedding gathers.
impl<Vocabulary: AxisName, Hidden: AxisName, Capacity: AxisName, Parameter, Chips: M + 'static>
    Op<Launch<'_, Hidden, Capacity, Chips>> for Embedding<Vocabulary, Hidden, Chips, Parameter>
where
    Parameter: Borrow<HbmTensor<bf16, Chips, m![Vocabulary, Hidden]>>,
{
    type Output = ();

    fn forward(
        &self,
        device: &mut Device,
        Launch { ids, count, output }: Launch<'_, Hidden, Capacity, Chips>,
    ) {
        type Span = axis::Rows<SPAN>;
        type Block = axis::Rows<BLOCK>;
        type Rest = axis::Rows<REST>;

        // Rows per slice of a span's half: at least two, over a power-of-two slice count.
        const P: usize = {
            let mut slices = 1;
            while slices < 256 && SPAN / 2 % (2 * slices) == 0 && SPAN / 2 / (2 * slices) >= 2 {
                slices *= 2;
            }
            SPAN / 2 / slices
        };
        let offsets = Op::forward(
            &ByteOffset::<bf16, m![Hidden], Packets<Capacity>>::default(),
            &mut *device,
            ids,
        );
        let weight: &HbmTensor<bf16, Chips, m![Vocabulary, Hidden]> = self.weight.borrow();
        let live = count.rows();
        for span in 0..live / SPAN {
            let offset = offsets.view().tile::<m![Capacity / Span], 1,
                m![1, Capacity / Span = 1 # (Capacity / Span), Capacity % Span]
            >(span);
            // SAFETY: the selected window owns exactly a span of consecutive row indices.
            let offset = unsafe { offset.alias::<Chips, m![Capacity % Span]>() };
            // SAFETY: one index per row, in row order.
            let offset = unsafe { offset.reshape::<Chips, m![Span]>() };
            // Whole rows move as one DMA element each, where feature tiles chain an index pass
            // and a gather per tile ahead of the first layer.
            let whole: DmTensor<
                bf16,
                Chips,
                m![Span / (Span / 2)],
                m![Span % (Span / 2) / { P } # 256],
                m![Span % { P }, Hidden],
            > = weight
                .gather::<m![Vocabulary], m![Hidden]>()
                .by_byte_offsets(&offset)
                .to_dm(&mut device.tdma);
            // Feature chunks lead each slice, so the relayout moves every chunk's rows as one run.
            let embedded: DmTensor<
                bf16,
                Chips,
                m![Span / (Span / 2)],
                m![Span % (Span / 2) / { P } # 256],
                m![Hidden / 16, Span % { P }, Hidden % 16],
            > = device
                .main
                .begin(whole.view())
                .fetch::<m![Hidden / 16, Span % { P }], m![Hidden % 16]>()
                .collect::<m![Hidden / 16, Span % { P }], m![Hidden % 16]>()
                .commit_trim::<m![Hidden % 16]>()
                .commit();
            let mut rows = Owned::<Hidden, Span, Chips>::new();
            embedded
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
            rows.view().to_hbm_view(
                &mut device.tdma,
                output
                    .view_mut()
                    .tile::<m![{ Symbol<Capacity> }], { SPAN }, m![
                        Hidden / 16,
                        { Symbol<Capacity> } = { SPAN } #{!} ({ Symbol<Capacity> }),
                        Hidden % 16
                    ]>(span * SPAN),
            );
        }
        // The block gathers after the whole spans; the difference saturates at zero blocks.
        let first = live / SPAN * (SPAN / BLOCK);
        for block in 0..live / BLOCK + (live % BLOCK + BLOCK - REST - 1) / BLOCK - first {
            let block = first + block;
            let offset = offsets.view().tile::<m![Capacity / Block], 1,
                m![1, Capacity / Block = 1 # (Capacity / Block), Capacity % Block]
            >(block);
            // SAFETY: the selected window owns exactly a block of consecutive row indices.
            let offset = unsafe { offset.alias::<Chips, m![Capacity % Block]>() };
            // SAFETY: packet groups reinterpret the same indices without changing their order.
            let offset = unsafe { offset.reshape::<Chips, Packets<Block>>() };
            let embedded = self.tiles::<Block, _>(&mut *device, &offset);
            let mut rows: DmTensor<
                bf16,
                Chips,
                m![Capacity % Block / (Block / 2)],
                m![Hidden / 16 # 256],
                m![Capacity % (Block / 2), Hidden % 16],
            > = DmTensor::new();
            embedded
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
            rows.view().to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Capacity / Block], 1, m![
                    Hidden / 16, Capacity / Block = 1 #{!} (Capacity / Block),
                    Capacity % Block, Hidden % 16,
                ]>(block),
            );
        }
        for _ in 0..(live % BLOCK + BLOCK - REST - 1) % BLOCK / (BLOCK - REST) {
            let block = BLOCK / REST * (live / BLOCK);
            let offset = offsets
                .view()
                .tile::<
                    m![Capacity / Rest],
                    1,
                    m![1, Capacity / Rest = 1 # (Capacity / Rest), Capacity % Rest],
                >(
                    block,
                );
            // SAFETY: the selected window owns exactly a rest of consecutive row indices.
            let offset = unsafe { offset.alias::<Chips, m![Capacity % Rest]>() };
            // SAFETY: one packet group reinterprets the same indices in order.
            let offset = unsafe { offset.reshape::<Chips, Packets<Rest>>() };
            let embedded = self.tiles::<Rest, _>(&mut *device, &offset);
            let mut rows: DmTensor<
                bf16,
                Chips,
                m![1 # 2],
                m![Hidden / 16 # 256],
                m![Capacity % Rest, Hidden % 16],
            > = DmTensor::new();
            embedded
                .view()
                .to_dm_view(&mut device.tdma, rows.view_mut());
            rows.view().to_hbm_view(
                &mut device.tdma,
                output.view_mut().tile::<m![Capacity / Rest], 1, m![
                    Hidden / 16, Capacity / Rest = 1 #{!} (Capacity / Rest), Capacity % Rest,
                    Hidden % 16,
                ]>(block),
            );
        }
    }
}

/// A launch's embedded rows, its tokens and live rows read from the env: the node's first poll
/// embeds every live row into its slot, the launch's hidden rows, and each poll gives a span.
impl<
    'm,
    's,
    'i,
    U,
    E,
    Vocabulary: AxisName,
    Hidden: AxisName,
    Capacity: AxisName,
    Chips: M + 'static,
> IntoStream<U, E, &'s mut HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>>
    for &'m Embedding<Vocabulary, Hidden, Chips>
where
    U: Stream,
    E: Copy + Into<(&'i Ids<Capacity, Chips>, Live<(), Capacity>)>,
{
    type Stream = Node<'m, 's, Vocabulary, Hidden, Capacity, E, Chips>;

    fn into_stream(
        self,
        env: E,
        rows: &'s mut HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>,
        _: U,
    ) -> Self::Stream {
        Node {
            embedding: self,
            env,
            rows,
            given: 0,
        }
    }
}

/// An embedding's node: the env it reads the launch from, the rows it embeds into and the spans
/// of them it gave so far.
pub struct Node<
    'm,
    's,
    Vocabulary: AxisName,
    Hidden: AxisName,
    Capacity: AxisName,
    E,
    Chips: M + 'static = Chip,
> {
    embedding: &'m Embedding<Vocabulary, Hidden, Chips>,
    env: E,
    rows: &'s mut HbmTensor<bf16, Chips, Blocked<Hidden, Capacity>>,
    given: usize,
}

impl<'i, Vocabulary: AxisName, Hidden: AxisName, Capacity: AxisName, E, Chips: M + 'static> Stream
    for Node<'_, '_, Vocabulary, Hidden, Capacity, E, Chips>
where
    E: Copy + Into<(&'i Ids<Capacity, Chips>, Live<(), Capacity>)>,
{
    type Item = Live<Owned<Hidden, span::Span, Chips>, Capacity>;

    fn poll_next(&mut self, device: &mut Device) -> Poll<Option<Self::Item>> {
        let (ids, live) = self.env.into();
        if self.given == 0 {
            let launch = Launch {
                ids,
                count: live,
                output: &mut *self.rows,
            };
            Op::forward(self.embedding, &mut *device, launch);
        }
        if self.given < live.count::<span::Span>() {
            let tile = live.tile::<span::Span>(self.given);
            self.given += 1;
            return Poll::Ready(Some(tile.with(self.rows.fill(device, tile))));
        }
        Poll::Ready(None)
    }
}

/// Places embedding rows in the feature mapping required by a decoder.
#[derive(DeviceSend)]
pub struct Reorder<Hidden>(PhantomData<Hidden>);

impl<Hidden> Default for Reorder<Hidden> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<Hidden: AxisName, Rows: AxisName, Env: Copy, Chips: M + 'static>
    Module<Vectors<Hidden, Rows, Chips>, Env, &mut ()> for Reorder<Hidden>
{
    type Output = Spread<Hidden, Rows, Chips>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: Vectors<Hidden, Rows, Chips>,
    ) -> Self::Output {
        mapping::Reorder::default().forward(device, env, &mut (), input)
    }
}

/// Only the live rows: every row in one copy when every packet is live, else each serial step of
/// the gathered rows that holds a live row; a later step's rows are left unwritten.
impl<Hidden: AxisName, Rows: AxisName, Env: Copy, Chips: M + 'static>
    Module<Live<Vectors<Hidden, Rows, Chips>, Rows>, Env, &mut ()> for Reorder<Hidden>
{
    type Output = Live<Spread<Hidden, Rows, Chips>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        env: Env,
        _: &mut (),
        input: Live<Vectors<Hidden, Rows, Chips>, Rows>,
    ) -> Self::Output {
        let live = input.with(());
        let value = input.into_value();
        if live.count::<Lanes<Rows>>() == <Groups<Rows> as M>::SIZE {
            return live.with(Module::forward(&self, device, env, &mut (), value));
        }
        // Gathered rows split each step of `Lanes * Clusters` rows over clusters and slices, so a
        // step is the smallest copy.
        let step = Lanes::<Rows>::SIZE * Clusters::<Rows>::SIZE;
        let mut output: Spread<Hidden, Rows, Chips> = DmTensor::new();
        for serial in 0..(live.rows() + step - 1) / step {
            value
                .view()
                .tile::<Layout<Rows>::Serial, 1, m![
                    Hidden / 512,
                    { Layout<Rows>::Serial } = 1 # ({ Layout<Rows>::Serial }),
                    Hidden % 16
                ]>(serial)
                .to_dm_view(
                    &mut device.tdma,
                    output.view_mut().tile::<Layout<Rows>::Serial, 1, m![
                        { Layout<Rows>::Serial } = 1 #{!} ({ Layout<Rows>::Serial }),
                        { Layout<Rows>::Spatial },
                        { Layout<Rows>::Lanes },
                        Hidden % ({ Symbol<Chunk<Hidden>> })
                    ]>(serial),
                );
        }
        live.with(output)
    }
}

/// A launch's supplied rows in the decoder's feature mapping, with the launch's live rows.
impl<'x, Hidden: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<(&'x HbmTensor<bf16, Chips, m![Rows, Hidden]>, Live<(), Rows>)> for Reorder<Hidden>
{
    type Output = Live<Spread<Hidden, Rows, Chips>, Rows>;

    fn forward(
        &self,
        device: &mut Device,
        (rows, tiles): (&'x HbmTensor<bf16, Chips, m![Rows, Hidden]>, Live<(), Rows>),
    ) -> Self::Output {
        tiles.with(rows.to_dm(&mut device.tdma))
    }
}

/// Embedding parameters whose row and column counts come from the checkpoint.
#[derive(DeviceSend)]
pub struct Lookup<Weight> {
    pub weight: Weight,
}

impl<Weight> Lookup<Weight> {
    pub fn new(weight: Weight) -> Self {
        Self { weight }
    }
}

impl<D: Scalar, Vocabulary: AxisName, Width: AxisName, Rows: AxisName, Chips: M + 'static>
    Op<&HbmTensor<i32, Chips, m![Rows]>> for Lookup<HbmTensor<D, Chips, m![Vocabulary, Width]>>
{
    type Output = HbmTensor<D, Chips, m![Rows, Width]>;

    fn forward(&self, _: &mut Device, _: &HbmTensor<i32, Chips, m![Rows]>) -> Self::Output {
        todo!("NPU embedding lookup into contiguous rows")
    }
}

/// A factorized position embedding: a table row for each position along each of two coordinates,
/// stored as one tensor under the embedding's own name.
#[derive(DeviceSend, Load)]
pub struct Factorized<Table> {
    #[load(rename = "")]
    pub table: Table,
}

/// Each item's position row: the sum of its coordinates' table rows, read at the rows' byte
/// offsets, in FP32 and rounded once.
impl<'x, 't, P: M, W: AxisName, R: AxisName, D: Float, Chips: M + 'static>
    Op<(
        &'x HbmTensor<i32, Chips, m![R]>,
        &'x HbmTensor<i32, Chips, m![R]>,
    )> for Factorized<&'t HbmTensor<D, Chips, m![{ P }, W]>>
{
    type Output = HbmTensor<D, Chips, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        (first, second): (
            &'x HbmTensor<i32, Chips, m![R]>,
            &'x HbmTensor<i32, Chips, m![R]>,
        ),
    ) -> Self::Output {
        let first = Op::forward(&Gather, &mut *device, (self.table, first));
        let second = Op::forward(&Gather, &mut *device, (self.table, second));
        Op::forward(&Add, device, (&first, &second))
    }
}

/// The owned table runs as its borrowed view.
impl<'x, P: M, W: AxisName, R: AxisName, D: Float, Chips: M + 'static>
    Op<(
        &'x HbmTensor<i32, Chips, m![R]>,
        &'x HbmTensor<i32, Chips, m![R]>,
    )> for Factorized<HbmTensor<D, Chips, m![{ P }, W]>>
{
    type Output = HbmTensor<D, Chips, m![R, W]>;

    fn forward(
        &self,
        device: &mut Device,
        offsets: (
            &'x HbmTensor<i32, Chips, m![R]>,
            &'x HbmTensor<i32, Chips, m![R]>,
        ),
    ) -> Self::Output {
        Op::forward(&Factorized { table: &self.table }, device, offsets)
    }
}

#[cfg(test)]
mod tests {
    use furiosa_opt_std::prelude::*;
    use furiosa_opt_std::runtime::launch;

    use crate::ops::normalization::{Blocked, RowMajor};
    use crate::tensor::axis::{Eight, Single, Sixteen};
    use crate::tensor::form::Weight;
    use crate::tensor::live::Live;
    use crate::tensor::parallel::{Parallelism, Replicated};
    use crate::tensor::{live::tests::count, scalar::Value};
    use crate::{Module, Op};

    use super::{Embedding, Ids};
    use crate::tensor::mapping::Packets;
    use std::io;
    use std::marker::PhantomData;

    axes![Table = 64, Served = 1024, Wide = 5120];

    type Chip = m![1];

    type Two = <Replicated<2> as Parallelism>::Chip;

    axes![Model = 4096];

    /// A launch's rows: the span gathers' rows, as every launch names them.
    type Capacity = crate::tensor::axis::Rows<1024>;

    #[device(chip = 1)]
    fn forward<Vocabulary: AxisName, Hidden: AxisName, Rows: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocabulary, Hidden>,
        ids: &Ids<Rows>,
        output: &mut HbmTensor<bf16, m![1], RowMajor<Hidden, Rows>>,
    ) {
        Op::forward(embedding, &mut *device, ids)
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    async fn embedding<Vocabulary: AxisName, Hidden: AxisName>(
        device: &mut Device,
        weight: HostTensor<bf16, m![Vocabulary, Hidden]>,
    ) -> io::Result<Embedding<Vocabulary, Hidden>> {
        Ok(Embedding {
            features: PhantomData,
            weight: Weight::new(
                weight
                    .to_hbm(&mut device.pdma)
                    .await
                    .map_err(io::Error::other)?,
            ),
        })
    }

    fn table<Hidden: AxisName>() -> Vec<bf16> {
        (0..Table::SIZE * Hidden::SIZE)
            .map(|index| bf16::from_f32(((index * 37) % 101) as f32 / 25.0 - 2.0))
            .collect()
    }

    /// The token ids, eight to the packet, the lanes past them repeating
    /// the first.
    fn ids(rows: &[usize]) -> Vec<i32> {
        (0..8)
            .map(|lane| rows[lane.min(rows.len() - 1)] as i32)
            .collect()
    }

    /// Eight tokens' rows are written each in its own hidden row, bit for bit.
    #[tokio::test]
    async fn gathers_a_row_per_token() {
        let mut device = Device::new(forward.topology()).unwrap();
        let values = table::<Served>();
        let rows = [23, 5, 61, 0, 40, 7, 63, 12];
        let embedding =
            embedding::<Table, Served>(&mut device, HostTensor::from_vec(values.clone()))
                .await
                .unwrap();
        let offsets = HostTensor::<i32, Packets<Eight>>::from_vec(ids(&rows))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: HbmTensor<bf16, Chip, RowMajor<Served, Eight>> =
            HostTensor::<bf16, RowMajor<Served, Eight>>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(forward, (&mut device, &embedding, &offsets, &mut output))
            .await
            .unwrap();

        let gathered = output
            .to_host::<RowMajor<Served, Eight>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, row) in rows.into_iter().enumerate() {
            assert_eq!(
                gathered[at * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                values[row * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "token {at} (row {row})"
            );
        }
    }

    #[device(chip = 2)]
    fn forward_two<Vocabulary: AxisName, Hidden: AxisName, Rows: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocabulary, Hidden, Two>,
        ids: &Ids<Rows, Two>,
        output: &mut HbmTensor<bf16, Two, RowMajor<Hidden, Rows>>,
    ) {
        Op::forward(embedding, &mut *device, ids)
            .view()
            .to_hbm_view(&mut device.tdma, output.view_mut());
    }

    /// On two chips each holding the table and the ids, each chip gathers the rows one chip does.
    #[tokio::test]
    async fn gathers_on_every_chip_of_a_placement() {
        let mut device = Device::new(forward_two.topology()).unwrap();
        let values = table::<Served>();
        let rows = [23, 5, 61, 0, 40, 7, 63, 12];
        let embedding = Embedding::<Table, Served, Two> {
            features: PhantomData,
            weight: Weight::new(
                HostTensor::<bf16, m![{ Two }, Table, Served]>::from_vec(values.repeat(2))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
        };
        let offsets =
            HostTensor::<i32, m![{ Two }, { Packets<Eight> }]>::from_vec(ids(&rows).repeat(2))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let mut output: HbmTensor<bf16, Two, RowMajor<Served, Eight>> =
            HostTensor::<bf16, m![{ Two }, { RowMajor<Served, Eight> }]>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(
            forward_two,
            (&mut device, &embedding, &offsets, &mut output),
        )
        .await
        .unwrap();

        let gathered = output
            .to_host::<m![{ Two }, { RowMajor<Served, Eight> }]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (chip, gathered) in gathered
            .chunks_exact(Eight::SIZE * Served::SIZE)
            .enumerate()
        {
            for (at, row) in rows.into_iter().enumerate() {
                assert_eq!(
                    bits(&gathered[at * Served::SIZE..][..Served::SIZE]),
                    bits(&values[row * Served::SIZE..][..Served::SIZE]),
                    "chip {chip} token {at} (row {row})"
                );
            }
        }
    }

    #[device(chip = 2)]
    fn launch_two<Vocabulary: AxisName, Hidden: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocabulary, Hidden, Two>,
        ids: &Ids<Capacity, Two>,
        rows: &Value<Two>,
        output: &mut HbmTensor<bf16, Two, Blocked<Hidden, Capacity>>,
    ) {
        let count = Live::<(), Capacity>::read(rows);
        Op::forward(
            embedding,
            &mut *device,
            super::Launch { ids, count, output },
        );
    }

    /// On two chips each holding the table and a launch's ids, the live rows hold their table rows
    /// on each chip bit for bit: 645 rows of a model's width, a gather of a whole span, one of a
    /// block and a short rest.
    #[tokio::test]
    async fn launches_on_every_chip_of_a_placement() {
        let mut device = Device::new(launch_two.topology()).unwrap();
        let values = table::<Model>();
        let tokens: Vec<i32> = (0..Capacity::SIZE)
            .map(|row| ((row * 37 + 11) % Table::SIZE) as i32)
            .collect();
        let live_rows = 512 + 128 + 5;
        let embedding = Embedding::<Table, Model, Two> {
            features: PhantomData,
            weight: Weight::new(
                HostTensor::<bf16, m![{ Two }, Table, Model]>::from_vec(values.repeat(2))
                    .to_hbm(&mut device.pdma)
                    .await
                    .unwrap(),
            ),
        };
        let ids = HostTensor::<i32, m![{ Two }, { Packets<Capacity> }]>::from_vec(tokens.repeat(2))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rows = HostTensor::<i32, m![{ Two }, 1]>::from_vec([live_rows as i32; 2])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: HbmTensor<bf16, Two, Blocked<Model, Capacity>> =
            HostTensor::<bf16, m![{ Two }, { Blocked<Model, Capacity> }]>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(
            launch_two,
            (&mut device, &embedding, &ids, &rows, &mut output),
        )
        .await
        .unwrap();

        let output = output
            .to_host::<m![{ Two }, { RowMajor<Model, Capacity> }]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        let bits = |values: &[bf16]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        for (chip, copy) in output
            .chunks_exact(Capacity::SIZE * Model::SIZE)
            .enumerate()
        {
            for (at, &token) in tokens.iter().take(live_rows).enumerate() {
                let row = token as usize;
                assert_eq!(
                    bits(&copy[at * Model::SIZE..][..Model::SIZE]),
                    bits(&values[row * Model::SIZE..][..Model::SIZE]),
                    "chip {chip} token {at} (row {row})"
                );
            }
        }
    }

    /// A 5120-wide row spans 320 slices, exceeding one gather instruction's slice count.
    #[tokio::test]
    async fn gathers_a_width_the_monolithic_geometry_cannot_lay_out() {
        let mut device = Device::new(forward.topology()).unwrap();
        let values = table::<Wide>();
        let row = 41;
        let embedding = embedding::<Table, Wide>(&mut device, HostTensor::from_vec(values.clone()))
            .await
            .unwrap();
        let offsets = HostTensor::<i32, Packets<Single>>::from_vec(ids(&[row]))
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output: HbmTensor<bf16, Chip, RowMajor<Wide, Single>> =
            HostTensor::<bf16, RowMajor<Wide, Single>>::zero()
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();

        launch(forward, (&mut device, &embedding, &offsets, &mut output))
            .await
            .unwrap();

        let gathered = output
            .to_host::<RowMajor<Wide, Single>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (column, observed) in gathered.iter().enumerate() {
            let expected = values[row * Wide::SIZE + column];
            assert_eq!(
                observed.to_bits(),
                expected.to_bits(),
                "column {column}: expected {}, observed {}",
                expected.to_f32(),
                observed.to_f32(),
            );
        }
    }

    #[device(chip = 1)]
    fn live<Vocabulary: AxisName, Hidden: AxisName>(
        device: &mut Device,
        embedding: &Embedding<Vocabulary, Hidden>,
        rows: &Value,
        ids: &Ids<Sixteen>,
    ) -> HbmTensor<bf16, Chip, RowMajor<Hidden, Sixteen>> {
        Module::forward(
            &embedding,
            &mut *device,
            (),
            &mut (),
            (ids, Live::<(), Sixteen>::read(rows)),
        )
        .value()
        .to_hbm(&mut device.tdma)
    }

    /// The live tokens of a partial launch, one whole row group and part of the next, each copy
    /// their table row bit for bit in the decoder's feature mapping.
    #[tokio::test]
    async fn writes_the_rows_of_live_tokens() {
        let mut device = Device::new(live.topology()).unwrap();
        let values = table::<Served>();
        let tokens: Vec<usize> = (0..Sixteen::SIZE)
            .map(|row| (row * 37 + 11) % Table::SIZE)
            .collect();
        let embedding =
            embedding::<Table, Served>(&mut device, HostTensor::from_vec(values.clone()))
                .await
                .unwrap();
        let ids =
            HostTensor::<i32, Packets<Sixteen>>::from_vec(tokens.iter().map(|&row| row as i32))
                .to_hbm(&mut device.pdma)
                .await
                .unwrap();
        let live_rows = 11;
        let rows = count(live_rows).to_hbm(&mut device.pdma).await.unwrap();
        let output = launch(live, (&mut device, &embedding, &rows, &ids))
            .await
            .unwrap()
            .to_host::<RowMajor<Served, Sixteen>>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for (at, row) in tokens.into_iter().take(live_rows).enumerate() {
            assert_eq!(
                output[at * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                values[row * Served::SIZE..][..Served::SIZE]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "token {at} (row {row})"
            );
        }
    }
}
