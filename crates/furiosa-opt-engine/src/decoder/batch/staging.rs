//! A call's operands in device buffers of its own: its compact history, live row count and
//! extra read arrays in one buffer and one transfer, and its token rows at the compiled envelope.

use super::{Batch, Rows, compact::Table};
use furiosa_opt_models::modules::attention::key_value::History;
use furiosa_opt_models::ops::index::{copy, table::FIELDS};
use furiosa_opt_models::tensor::{memory::alias, scalar::Value};
use furiosa_opt_std::prelude::*;
use std::{borrow::Cow, io, ops::Range};

/// A call's staged operands for at most `R` rows, a copy on each chip of `Chips`: its history of
/// block tables of `B` words, its live row count and its token rows.
pub struct Staged<R: AxisName, B: AxisName, Chips: M> {
    pub history: History<R, B, R, Chips>,
    pub count: Value<Chips>,
    /// One I32 token ID for each of the `R` rows, which the call's entry reads; rows past the
    /// live ones hold token 0.
    pub tokens: HbmBuffer,
}

impl<R: AxisName, B: AxisName, Chips: M> Staged<R, B, Chips> {
    /// Words of the history's segments, blocks, selected rows and extents at their compiled
    /// capacities, then of the live count, in the one buffer a call stages.
    const fn capacities() -> [usize; 5] {
        [R::SIZE * FIELDS, B::SIZE, R::SIZE, 3, 1]
    }
}

impl<A> Batch<'_, A> {
    /// Stages the call for at most `R` rows, a copy on every chip of `Chips`, and `reads` (values
    /// and capacities in words) in one transfer, with a view of each; `table` forms the history of
    /// cache group `group`.
    pub async fn stage<R: AxisName, B: AxisName, Chips: M>(
        &self,
        device: &mut Device,
        reads: &[(Vec<i32>, usize)],
        (group, table): (usize, Table),
    ) -> io::Result<(Staged<R, B, Chips>, Vec<HbmBuffer>)> {
        let compact = &self.compact[group];
        let history = compact.arrays::<R, B, R>(table)?;
        if compact.rows() > R::SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "live rows exceed the compiled envelope",
            ));
        }
        let live = [i32::try_from(compact.rows()).map_err(io::Error::other)?];
        let staging = Staging::new(
            (history.iter().map(Vec::as_slice).chain([&live[..]]))
                .zip(Staged::<R, B, Chips>::capacities()),
            reads.iter().map(|(values, words)| (&values[..], *words)),
        );
        let bytes = R::SIZE * Rows::<i32, m![1]>::bytes();
        // A lone request of the envelope's rows is read in place, sparing a copy launch.
        let lone = match &self.tokens[..] {
            [(tokens, _)] if tokens.buffer().size() == bytes => {
                Some(tokens.buffer().alias(..).map_err(io::Error::other)?)
            }
            _ => None,
        };
        // The copy launch that places several requests' rows runs on one chip.
        if lone.is_none() && Chips::SIZE > 1 {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "a call on several chips reads one request of the compiled rows in place",
            ));
        }
        let mut target = match lone {
            Some(_) => None,
            None => Some(HbmBuffer::alloc(device, bytes).map_err(io::Error::other)?),
        };
        // A row past the call's reads token 0: the lookup gathers every row's embedding.
        let buffer = staging.to_hbm::<Chips>(device, target.as_mut()).await?;
        let mut ranges = staging.ranges.into_iter();
        let [segments, blocks, selected, extents, count] =
            <[_; 5]>::try_from(ranges.by_ref().take(5).collect::<Vec<_>>())
                .unwrap_or_else(|_| unreachable!("the history stages five arrays"));
        let history = History {
            segments: alias(&buffer, segments)?,
            blocks: alias(&buffer, blocks)?,
            selected: alias(&buffer, selected)?,
            extents: alias(&buffer, extents)?,
        };
        let count = alias(&buffer, count)?;
        let views = ranges
            .map(|range| buffer.alias(range).map_err(io::Error::other))
            .collect::<io::Result<_>>()?;
        let tokens = match target {
            Some(target) => {
                let bytes = Rows::<i32, m![1]>::bytes();
                let mut targets = (self.tokens.iter())
                    .map(|(_, rows)| target.alias(rows.start * bytes..rows.end * bytes))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(io::Error::other)?;
                let sources = self.tokens.iter().map(|(tokens, _)| tokens.buffer());
                copy(&mut *device, sources.zip(&mut targets))?;
                target
            }
            None => {
                lone.unwrap_or_else(|| unreachable!("a call without a copy target reads in place"))
            }
        };
        let staged = Staged {
            history,
            count,
            tokens,
        };
        Ok((staged, views))
    }
}

/// Bytes every array of a call's staging buffer starts at: a transfer from an array starting
/// mid-burst runs several times slower than one from an aligned start.
const ALIGNMENT: usize = 256;

/// Arrays of words a call stages in its own device buffer: an array the launch writes holds its
/// capacity alone, and each array it only reads starts past the previous one's values.
struct Staging {
    /// Each array's bytes in the buffer, the written arrays' first. A read array's capacity past
    /// its values overlaps the read arrays after it, which a launch reads only within their values.
    ranges: Vec<Range<usize>>,
    bytes: usize,
    /// The bytes the arrays' values fill, joined where only alignment parts them; `host` holds
    /// them in turn.
    filled: Vec<Range<usize>>,
    host: Vec<u8>,
}

impl Staging {
    /// The arrays a launch writes, then those it only reads, as values and capacities in words; no
    /// transfer writes the bytes past an array's values.
    fn new<'a>(
        written: impl IntoIterator<Item = (&'a [i32], usize)>,
        read: impl IntoIterator<Item = (&'a [i32], usize)>,
    ) -> Self {
        let arrays: Vec<_> = (written.into_iter().map(|array| (array, true)))
            .chain(read.into_iter().map(|array| (array, false)))
            .collect();
        let bytes = Self::extent(arrays.iter().map(|&((_, capacity), _)| capacity));
        let mut ranges = Vec::with_capacity(arrays.len());
        let mut filled: Vec<Range<usize>> = Vec::new();
        let mut host = Vec::new();
        let mut next = 0_usize;
        for ((values, capacity), alone) in arrays {
            assert!(
                values.len() <= capacity,
                "an array's values fit its capacity"
            );
            let start = next.next_multiple_of(ALIGNMENT);
            let range = start..start + capacity * size_of::<i32>();
            let end = range.start + size_of_val(values);
            next = if alone { range.end } else { end };
            ranges.push(range.clone());
            match filled.last_mut() {
                _ if values.is_empty() => continue,
                Some(last) if range.start - last.end < ALIGNMENT => {
                    host.resize(host.len() + range.start - last.end, 0);
                    last.end = end;
                }
                _ => filled.push(range.start..end),
            }
            let start = host.len();
            host.resize(start + size_of_val(values), 0);
            for (word, value) in host[start..].chunks_exact_mut(size_of::<i32>()).zip(values) {
                word.copy_from_slice(&value.to_le_bytes());
            }
        }
        Self {
            ranges,
            bytes,
            filled,
            host,
        }
    }

    /// Copies the values into a buffer of their own, a copy on each chip of `Chips`, with zeros
    /// over `zeroed` in the same transfer, which ends before the future does.
    async fn to_hbm<Chips: M>(
        &self,
        device: &mut Device,
        zeroed: Option<&mut HbmBuffer>,
    ) -> io::Result<HbmBuffer> {
        let buffer = HbmBuffer::alloc(device, self.bytes).map_err(io::Error::other)?;
        let mut targets = (self.filled.iter())
            .map(|range| buffer.alias(range.clone()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(io::Error::other)?;
        let zeros = vec![0; zeroed.as_ref().map_or(0, |target| target.size())];
        // A transfer to every chip of a buffer reads one copy of the bytes for each chip in turn.
        let copies: Vec<Cow<'_, [u8]>> = (self.filled.iter())
            .scan(&self.host[..], |host, range| {
                let (bytes, rest) = host.split_at(range.len());
                *host = rest;
                Some(match Chips::SIZE {
                    1 => Cow::Borrowed(bytes),
                    chips => Cow::Owned(bytes.repeat(chips)),
                })
            })
            .collect();
        let mut transfers = Vec::with_capacity(targets.len() + 1);
        for (target, bytes) in targets.iter_mut().zip(&copies) {
            let target = target.slice_mut(.., ..).map_err(io::Error::other)?;
            transfers.push((HostBufferSlice::from(&bytes[..]), target));
        }
        if let Some(target) = zeroed {
            let target = target.slice_mut(.., ..).map_err(io::Error::other)?;
            transfers.push((HostBufferSlice::from(&zeros[..]), target));
        }
        (device.pdma.to_hbm(transfers).await).map_err(io::Error::other)?;
        Ok(buffer)
    }

    /// Bytes of a buffer that holds arrays of `capacities` words, each from an aligned start past
    /// the capacity of the one before: the most any values of theirs take.
    fn extent(capacities: impl IntoIterator<Item = usize>) -> usize {
        (capacities.into_iter())
            .fold(0, |end: usize, capacity| {
                end.next_multiple_of(ALIGNMENT) + capacity * size_of::<i32>()
            })
            .next_multiple_of(ALIGNMENT)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::{Cache, CacheSpec, Group, Keep, Kind, Limits, Pool, Request};
    use super::*;
    use furiosa_opt_models::tensor::parallel::Chip;

    axes![Envelope = 8, Tables = 64];

    /// The I32 words of `buffer`, read back once the writes submitted before it completed.
    async fn words(device: &mut Device, buffer: &HbmBuffer) -> io::Result<Vec<i32>> {
        let mut host = vec![0; buffer.size()];
        (buffer.to_host(&mut device.pdma).output(&mut host[..]).await).map_err(io::Error::other)?;
        Ok((host.as_chunks::<4>().0.iter())
            .map(|&word| i32::from_le_bytes(word))
            .collect())
    }

    /// Each request's token IDs fill its call rows of the staged tokens, the rows past them hold
    /// token 0, the live count is the call's rows, and a read array's view leads with its values.
    #[test]
    fn stages_each_request_at_its_rows() -> io::Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let mut device =
                Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
            let limits = Limits {
                window: 64,
                rows: Envelope::SIZE,
                cache: CacheSpec {
                    pools: vec![Pool {
                        block: 16,
                        bytes: 64,
                    }],
                    groups: vec![Group {
                        pool: 0,
                        layers: 0..1,
                        kind: Kind::Paged,
                        window: None,
                    }],
                },
                adapters: 0,
            };
            let cache = Cache::new(&device, &limits.cache, 4)?;
            let mut requests = Vec::new();
            for (block, ids) in [[11, 12, 13].as_slice(), &[21, 22]].into_iter().enumerate() {
                let bytes: Vec<u8> = ids.iter().flat_map(|id: &i32| id.to_le_bytes()).collect();
                let tokens = (HostBufferSlice::from(&bytes[..])
                    .to_hbm(&mut device.pdma)
                    .await)
                    .map_err(io::Error::other)?;
                requests.push(Request {
                    rows: vec![vec![u32::try_from(block).unwrap()]],
                    cached: 0,
                    tokens: Rows::new(&tokens, 0..ids.len())?,
                    keep: Keep::Last,
                    adapter: None,
                    positions: None,
                    mask: None,
                    overrides: Vec::new(),
                });
            }
            let batch = Batch::<()>::new(&requests, &limits, &cache, &[])?;
            let read = vec![5, 6, 7];
            let (staged, views) = batch
                .stage::<Envelope, Tables, Chip>(
                    &mut device,
                    &[(read.clone(), 70)],
                    (0, Table::Capacity),
                )
                .await?;
            let tokens = words(&mut device, &staged.tokens).await?;
            assert_eq!(tokens, [11, 12, 13, 21, 22, 0, 0, 0]);
            let count = words(&mut device, &staged.count.into_buffer()).await?;
            assert_eq!(count, [5]);
            let view = words(&mut device, &views[0]).await?;
            assert_eq!(view[..read.len()], read);
            Ok(())
        })
    }
}
