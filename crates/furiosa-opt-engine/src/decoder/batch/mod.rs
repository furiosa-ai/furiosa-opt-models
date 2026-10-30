//! One call of a decoder: its requests validated and planned on the host, then its operands
//! staged on the device.
//!
//! - `Batch::new` is the only place a request is validated. It reads no token ID: the IDs stay on
//!   the device, and the framework keeps each one in the vocabulary.
//! - The call's rows are the appended rows of its one-row requests, then of the others, each in
//!   request order; its compact history names each request's rows, its cached positions and the
//!   blocks they use. A call returns its kept rows in request order.

mod compact;
mod operands;
mod staging;

pub use compact::Table;
pub use staging::Staged;

use super::{Binding, Cache, Group, Keep, Kind, Limits, Request};
use furiosa_opt_models::ops::index::table::FIELDS;
use furiosa_opt_std::prelude::*;
use std::{collections::HashSet, io, marker::PhantomData, ops::Range};

/// Rotary positions are F32 values, exact below this.
const POSITIONS: usize = 1 << 24;

/// Device rows of `Row`-mapped `D` elements stored one after another, as many as run time gives:
/// a request's tokens, or the rows a call returns.
pub struct Rows<D, Row> {
    /// Exactly the rows' bytes.
    buffer: HbmBuffer,
    row: PhantomData<fn() -> (D, Row)>,
}

impl<D, Row> Clone for Rows<D, Row> {
    fn clone(&self) -> Self {
        Self {
            buffer: (self.buffer.alias(..))
                .unwrap_or_else(|_| unreachable!("a buffer aliases itself")),
            row: PhantomData,
        }
    }
}

impl<D: Scalar, Row: M> Rows<D, Row> {
    /// Bytes of one row.
    pub fn bytes() -> usize {
        D::size_in_bytes_from_length(Row::SIZE)
    }

    /// Rows `rows` of the rows stored from the start of `buffer`, sharing storage; a reversed
    /// range, or one past the buffer, is invalid input.
    pub fn new(buffer: &HbmBuffer, rows: Range<usize>) -> io::Result<Self> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        if rows.start > rows.end {
            return Err(invalid("rows end before they start"));
        }
        let end = (rows.end.checked_mul(Self::bytes()))
            .filter(|&end| end <= buffer.size())
            .ok_or_else(|| invalid("rows exceed device storage"))?;
        Ok(Self {
            buffer: buffer
                .alias(rows.start * Self::bytes()..end)
                .map_err(io::Error::other)?,
            row: PhantomData,
        })
    }

    pub fn len(&self) -> usize {
        self.buffer.size() / Self::bytes()
    }

    pub fn buffer(&self) -> &HbmBuffer {
        &self.buffer
    }

    pub fn into_buffer(self) -> HbmBuffer {
        self.buffer
    }
}

/// The plan of one call: every request's appended rows, in request order, and the adapters `A`
/// they run with.
pub struct Batch<'a, A> {
    /// Each cache group's compact history, in spec order.
    compact: Vec<compact::Descriptor>,
    /// Each request's token IDs and the call rows they fill.
    tokens: Vec<(Rows<i32, m![1]>, Range<usize>)>,
    /// Each cache group's appended rows of every request in call row order: what a launch's
    /// attention front plans.
    appended: Vec<Vec<operands::Appended>>,
    /// The call rows the call returns, in order.
    kept: Vec<usize>,
    /// The call's adapters bound to its slots and rows; none when no row runs with one.
    adapters: Option<Binding<'a, A>>,
}

impl<'a, A> Batch<'a, A> {
    /// The call of `requests` on `cache` within `limits`, with `adapters`, the caller's list its
    /// requests name adapters in; refuses a call that breaks a rule, with the rule's own message,
    /// before anything moves to the device.
    pub fn new(
        requests: &[Request],
        limits: &Limits,
        cache: &Cache,
        adapters: &[Option<&'a A>],
    ) -> io::Result<Self> {
        let invalid = |message| io::Error::new(io::ErrorKind::InvalidInput, message);
        let unsupported = |message| io::Error::new(io::ErrorKind::Unsupported, message);
        if cache.spec() != &limits.cache {
            return Err(invalid("cache was allocated for another model"));
        }
        // A windowed group's table is indexed by position as a whole one is; the read drops the
        // blocks below a row's window, so the host checks every group alike.
        let blocks: Vec<usize> = (limits.cache.groups.iter())
            .map(|group| match group {
                Group {
                    pool,
                    kind: Kind::Paged,
                    ..
                } => Ok(limits.cache.pools[*pool].block),
                _ => Err(unsupported("a call plans paged cache groups")),
            })
            .collect::<io::Result<_>>()?;
        if blocks.is_empty() {
            return Err(unsupported("a call plans paged cache groups"));
        }
        if requests.is_empty() {
            return Err(invalid("call has no requests"));
        }
        let adapted = !adapters.is_empty() || requests.iter().any(|r| r.adapter.is_some());
        if adapted && limits.adapters == 0 {
            return Err(unsupported("adapters are not served"));
        }
        if adapters.len() > limits.adapters {
            return Err(invalid("call holds more adapters than compiled"));
        }
        let refusal = |request: &Request| -> Option<io::Error> {
            if request
                .adapter
                .is_some_and(|slot| adapters.get(slot).is_none_or(Option::is_none))
            {
                return Some(invalid("request names a slot that holds no adapter"));
            }
            if request.positions.is_some() {
                return Some(unsupported("explicit positions are not served"));
            }
            if request.mask.is_some() {
                return Some(unsupported("visibility masks are not served"));
            }
            if !request.overrides.is_empty() {
                return Some(unsupported("row overrides are not served"));
            }
            let len = request.tokens.len();
            let end = request.cached.saturating_add(len);
            if request.rows.len() != blocks.len() {
                return Some(invalid("request holds a block list for each cache group"));
            }
            if len == 0 {
                return Some(invalid("request has no tokens"));
            }
            if end > limits.window {
                return Some(invalid("request extends past the context window"));
            }
            if end > POSITIONS {
                return Some(invalid("position is not below 2^24"));
            }
            for (ids, block) in request.rows.iter().zip(&blocks) {
                let used = end.div_ceil(*block);
                if used > ids.len() {
                    return Some(invalid("request has fewer blocks than its positions need"));
                }
                if !ids[..used].iter().all(|&id| (id as usize) < cache.blocks()) {
                    return Some(invalid("block ID is outside the cache"));
                }
            }
            match request.keep {
                Keep::Range(from, to) if from > to || to > len => {
                    Some(invalid("kept rows lie outside the appended rows"))
                }
                _ => None,
            }
        };
        if let Some(error) = requests.iter().find_map(refusal) {
            return Err(error);
        }
        let spans = layout(requests.iter().map(|request| request.tokens.len()));
        let count = spans.iter().map(|span| span.end).max().unwrap_or(0);
        if count > limits.rows {
            return Err(invalid("call has more rows than compiled"));
        }
        let distinct = blocks.iter().enumerate().all(|(group, &block)| {
            let mut written = HashSet::new();
            requests.iter().all(|request| {
                let ids = &request.rows[group];
                (request.cached..request.cached + request.tokens.len())
                    .all(|p| written.insert(ids[p / block] as usize * block + p % block))
            })
        });
        if !distinct {
            return Err(invalid("two appended positions share a cache slot"));
        }
        let kept: Vec<usize> = (requests.iter().zip(&spans))
            .flat_map(|(request, span)| match request.keep {
                Keep::None => span.end..span.end,
                Keep::Last => span.end - 1..span.end,
                Keep::All => span.clone(),
                Keep::Range(from, to) => span.start + from..span.start + to,
            })
            .collect();
        let mut ascending = kept.clone();
        ascending.sort_unstable();
        let mut slots = vec![None; count];
        for (request, span) in requests.iter().zip(&spans) {
            slots[span.clone()].fill(request.adapter);
        }
        let groups = blocks.iter().enumerate().map(|(group, &block)| {
            let mut segments: Vec<compact::Segment<'_>> = (requests.iter().zip(&spans))
                .map(|(request, rows)| compact::Segment {
                    blocks: &request.rows[group][..(request.cached + rows.len()).div_ceil(block)],
                    cached: request.cached,
                    rows: rows.clone(),
                })
                .collect();
            segments.sort_by_key(|segment| segment.rows.start);
            let appended = (segments.iter())
                .map(|segment| operands::Appended {
                    rows: segment.rows.clone(),
                    cached: segment.cached,
                    blocks: segment.blocks.to_vec(),
                })
                .collect();
            Ok((
                compact::Descriptor::encode(&segments, &ascending)?,
                appended,
            ))
        });
        let (compact, appended) = groups.collect::<io::Result<Vec<_>>>()?.into_iter().unzip();
        Ok(Self {
            compact,
            appended,
            tokens: (requests.iter().zip(spans))
                .map(|(request, rows)| (request.tokens.clone(), rows))
                .collect(),
            kept,
            adapters: Binding::new(adapters, slots),
        })
    }

    /// The call rows the call returns, in order.
    pub fn kept(&self) -> &[usize] {
        &self.kept
    }

    /// The adapters the call runs, bound to its slots and rows; none when no row runs with one.
    pub fn adapters(&self) -> Option<&Binding<'a, A>> {
        self.adapters.as_ref()
    }

    /// Each call row as a segment of its own in cache group `group`, its history the request's
    /// cached positions and the call's rows before it, with the block IDs the segments name.
    pub fn row_segments(&self, group: usize) -> (Vec<[i32; FIELDS]>, &[i32]) {
        self.compact[group].row_segments()
    }
}

/// The call rows of requests of `lens` appended rows: the one-row requests' from row 0, then the
/// others', each in request order, since a launch attends its one-row requests first.
fn layout(lens: impl Iterator<Item = usize> + Clone) -> Vec<Range<usize>> {
    let order = (lens.clone().enumerate().filter(|&(_, len)| len == 1))
        .chain(lens.clone().enumerate().filter(|&(_, len)| len != 1));
    let mut spans = vec![0..0; lens.count()];
    let mut start = 0;
    for (request, len) in order {
        spans[request] = start..start + len;
        start += len;
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::super::{CacheSpec, Pool};
    use super::*;

    /// One paged group of one layer whose blocks hold `block` tokens, and no adapter.
    fn limits(block: usize) -> Limits {
        Limits {
            window: 64,
            rows: 8,
            cache: CacheSpec {
                pools: vec![Pool { block, bytes: 64 }],
                groups: vec![Group {
                    pool: 0,
                    layers: 0..1,
                    kind: Kind::Paged,
                    window: None,
                }],
            },
            adapters: 0,
        }
    }

    /// A plan of no adapters.
    fn plan(
        requests: &[Request],
        limits: &Limits,
        cache: &Cache,
    ) -> io::Result<Batch<'static, ()>> {
        Batch::new(requests, limits, cache, &[])
    }

    fn device() -> io::Result<Device> {
        Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)
    }

    /// `count` token IDs on the device; a plan reads none of them.
    fn ids(device: &Device, count: usize) -> io::Result<Rows<i32, m![1]>> {
        let bytes = count.max(1) * Rows::<i32, m![1]>::bytes();
        let buffer = HbmBuffer::alloc(device, bytes).map_err(io::Error::other)?;
        Rows::new(&buffer, 0..count)
    }

    /// The request appending `count` tokens after `cached` positions in `blocks`, keeping the last.
    fn request(
        device: &Device,
        blocks: &[u32],
        cached: usize,
        count: usize,
    ) -> io::Result<Request> {
        Ok(Request {
            rows: vec![blocks.to_vec()],
            cached,
            tokens: ids(device, count)?,
            keep: Keep::Last,
            adapter: None,
            positions: None,
            mask: None,
            overrides: Vec::new(),
        })
    }

    /// Rows view the storage they lie in: a range ending at its end is accepted; one past it, one
    /// whose bytes overflow and a reversed one are refused as invalid input.
    #[test]
    fn refuses_rows_past_their_storage() -> io::Result<()> {
        let device = device()?;
        let bytes = 4 * Rows::<i32, m![1]>::bytes();
        let buffer = HbmBuffer::alloc(&device, bytes).map_err(io::Error::other)?;
        assert_eq!(Rows::<i32, m![1]>::new(&buffer, 1..4)?.len(), 3);
        for (rows, reason) in [
            (2..5, "rows exceed device storage"),
            (0..usize::MAX, "rows exceed device storage"),
            (3..2, "rows end before they start"),
        ] {
            let error = Rows::<i32, m![1]>::new(&buffer, rows.clone())
                .err()
                .unwrap();
            let refusal = (io::ErrorKind::InvalidInput, reason.to_owned());
            assert_eq!((error.kind(), error.to_string()), refusal, "{rows:?}");
        }
        Ok(())
    }

    /// A request's blocks past those its positions use are not checked: a block ID outside the
    /// cache there, as a framework pads its block table, is accepted.
    #[test]
    fn accepts_any_block_past_the_used_ones() -> io::Result<()> {
        let device = device()?;
        let limits = limits(4);
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let requests = [request(&device, &[7, 5, 99], 3, 2)?];
        plan(&requests, &limits, &cache)?;
        Ok(())
    }

    /// A call over a whole-history group and a sliding-window group plans each group's history
    /// from its own blocks; a request must name blocks for every group.
    #[test]
    fn plans_every_cache_group() -> io::Result<()> {
        let device = device()?;
        let mut limits = limits(4);
        limits.cache.groups.push(Group {
            pool: 0,
            layers: 1..2,
            kind: Kind::Paged,
            window: Some(8),
        });
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let mut request = request(&device, &[7, 5], 3, 2)?;
        let error = plan(std::slice::from_ref(&request), &limits, &cache).err();
        assert_eq!(
            error.map(|error| error.kind()),
            Some(io::ErrorKind::InvalidInput)
        );
        request.rows.push(vec![1, 6]);
        let batch = plan(std::slice::from_ref(&request), &limits, &cache)?;
        axes![Segments = 1, Blocks = 64, Kept = 1];
        for (group, ids) in [(0, [7, 5]), (1, [1, 6])] {
            let [_, blocks, _, _] =
                batch.compact[group].arrays::<Segments, Blocks, Kept>(Table::Capacity)?;
            assert_eq!(blocks[..2], ids);
        }
        Ok(())
    }

    /// `Last` returns the row of a request's last position, `All` its rows in position order,
    /// `Range` the ones it names and `None` none, with the rows of earlier requests first.
    #[test]
    fn keeps_selected_rows_in_request_order() -> io::Result<()> {
        let device = device()?;
        let limits = limits(16);
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let keeping = |blocks: &[u32], cached, count, keep| -> io::Result<Request> {
            Ok(Request {
                keep,
                ..request(&device, blocks, cached, count)?
            })
        };
        let requests = [
            keeping(&[7], 5, 1, Keep::Last)?,
            keeping(&[2], 0, 2, Keep::None)?,
            keeping(&[5], 0, 2, Keep::All)?,
            keeping(&[6], 0, 3, Keep::Range(1, 3))?,
        ];
        let kept = [0, 3, 4, 6, 7];
        assert_eq!(plan(&requests, &limits, &cache)?.kept(), kept);
        Ok(())
    }

    /// One-row requests take the call's first rows and the others follow, each in request order.
    #[test]
    fn lays_out_one_row_requests_first() {
        let spans = layout([3, 1, 2, 1].into_iter());
        assert_eq!(spans, [2..5, 0..1, 5..7, 1..2]);
    }

    /// Each rule refuses a call that breaks only it, as invalid input with its own message; a kept
    /// range may be empty but not reversed.
    #[test]
    fn refuses_invalid_requests() -> io::Result<()> {
        let device = device()?;
        let limits = limits(16);
        let cache = Cache::new(&device, &limits.cache, 64)?;
        let other = Cache::new(&device, &self::limits(4).cache, 64)?;
        let request = |blocks: &[u32], cached, count| request(&device, blocks, cached, count);
        let short = Limits {
            window: 4,
            ..limits.clone()
        };
        let wide = Limits {
            window: (1 << 24) + 1,
            cache: CacheSpec {
                pools: vec![Pool {
                    block: 1 << 20,
                    bytes: 64,
                }],
                ..limits.cache.clone()
            },
            ..limits.clone()
        };
        let wide_cache = Cache::new(&device, &wide.cache, 64)?;
        let two = Request {
            rows: vec![vec![7], vec![8]],
            ..request(&[7], 0, 1)?
        };
        let past = Request {
            keep: Keep::Range(0, 2),
            ..request(&[7], 0, 1)?
        };
        let reversed = Request {
            keep: Keep::Range(1, 0),
            ..request(&[7], 0, 1)?
        };
        let cases = [
            (vec![], &limits, &cache, "call has no requests"),
            (
                vec![request(&[7], 0, 1)?],
                &limits,
                &other,
                "cache was allocated for another model",
            ),
            (
                vec![two],
                &limits,
                &cache,
                "request holds a block list for each cache group",
            ),
            (
                vec![request(&[7], 0, 0)?],
                &limits,
                &cache,
                "request has no tokens",
            ),
            (
                vec![request(&[7], 2, 3)?],
                &short,
                &cache,
                "request extends past the context window",
            ),
            (
                vec![request(&(0..17).collect::<Vec<_>>(), 1 << 24, 1)?],
                &wide,
                &wide_cache,
                "position is not below 2^24",
            ),
            (
                vec![request(&[7], 15, 2)?],
                &limits,
                &cache,
                "request has fewer blocks than its positions need",
            ),
            (
                vec![request(&[64], 0, 1)?],
                &limits,
                &cache,
                "block ID is outside the cache",
            ),
            (
                vec![past],
                &limits,
                &cache,
                "kept rows lie outside the appended rows",
            ),
            (
                vec![reversed],
                &limits,
                &cache,
                "kept rows lie outside the appended rows",
            ),
            (
                vec![request(&[7], 0, 9)?],
                &limits,
                &cache,
                "call has more rows than compiled",
            ),
            (
                vec![request(&[7], 0, 1)?, request(&[7], 0, 1)?],
                &limits,
                &cache,
                "two appended positions share a cache slot",
            ),
        ];
        for (requests, limits, cache, message) in cases {
            let error = plan(&requests, limits, cache).err().expect(message);
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{message}");
            assert_eq!(error.to_string(), message);
        }
        let empty = Request {
            keep: Keep::Range(1, 1),
            ..request(&[7], 0, 1)?
        };
        assert!(plan(&[empty], &limits, &cache)?.kept().is_empty());
        Ok(())
    }

    /// A request that needs what the call does not serve is refused as unsupported, with the
    /// feature's own message, and so is a cache without paged groups or with another kind of group.
    #[test]
    fn refuses_unserved_requests() -> io::Result<()> {
        let device = device()?;
        let limits = limits(16);
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let plain = || request(&device, &[7], 0, 1);
        let values = || HbmBuffer::alloc(&device, 4).map_err(io::Error::other);
        let cases = [
            (
                Request {
                    adapter: Some(0),
                    ..plain()?
                },
                "adapters are not served",
            ),
            (
                Request {
                    positions: Some(values()?),
                    ..plain()?
                },
                "explicit positions are not served",
            ),
            (
                Request {
                    mask: Some(values()?),
                    ..plain()?
                },
                "visibility masks are not served",
            ),
            (
                Request {
                    overrides: vec![(0, values()?)],
                    ..plain()?
                },
                "row overrides are not served",
            ),
        ];
        for (request, message) in cases {
            let error = plan(&[request], &limits, &cache).err().expect(message);
            assert_eq!(error.kind(), io::ErrorKind::Unsupported, "{message}");
            assert_eq!(error.to_string(), message);
        }
        let planned = |edit: &dyn Fn(&mut Vec<Group>)| -> io::Result<io::Error> {
            let mut layered = limits.clone();
            edit(&mut layered.cache.groups);
            let cache = Cache::new(&device, &layered.cache, 8)?;
            Ok(plan(&[plain()?], &layered, &cache).err().expect("a group"))
        };
        for (edit, case) in [
            (
                &(|groups: &mut Vec<Group>| groups[0].kind = Kind::State)
                    as &dyn Fn(&mut Vec<Group>),
                "a state",
            ),
            (
                &|groups: &mut Vec<Group>| groups[0].kind = Kind::Cross { length: 4 },
                "an encoder",
            ),
            (&|groups: &mut Vec<Group>| groups.clear(), "no group"),
        ] {
            let error = planned(edit)?;
            let refusal = (error.kind(), error.to_string());
            let expected = (
                io::ErrorKind::Unsupported,
                "a call plans paged cache groups".into(),
            );
            assert_eq!(refusal, expected, "{case}");
        }
        Ok(())
    }

    /// A call with adapters refuses a list longer than the call holds and a request that names a
    /// slot outside the list or empty, each as invalid input with its own message.
    #[test]
    #[ignore = "needs a device: run on a chip with --ignored"]
    fn refuses_unheld_adapters() -> io::Result<()> {
        let device = device()?;
        let limits = Limits {
            adapters: 2,
            ..limits(16)
        };
        let cache = Cache::new(&device, &limits.cache, 8)?;
        let naming = |slot| -> io::Result<Request> {
            Ok(Request {
                adapter: Some(slot),
                ..request(&device, &[7], 0, 1)?
            })
        };
        let adapter = ();
        let cases = [
            (
                naming(0)?,
                vec![Some(&adapter); 3],
                "call holds more adapters than compiled",
            ),
            (
                naming(1)?,
                vec![Some(&adapter), None],
                "request names a slot that holds no adapter",
            ),
            (
                naming(2)?,
                vec![Some(&adapter); 2],
                "request names a slot that holds no adapter",
            ),
        ];
        for (request, adapters, message) in cases {
            let error = Batch::new(&[request], &limits, &cache, &adapters)
                .err()
                .expect(message);
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput, "{message}");
            assert_eq!(error.to_string(), message);
        }
        Ok(())
    }
}
