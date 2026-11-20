//! Checked valid-row transfers into head-major attention keys and values.
//!
//! Model geometry supplies runs in destination order. Each run copies only its valid source
//! rows; checked destination offsets are transported explicitly. The same request plan is
//! reused for K and V.

use super::{Copy as CopyRows, KeyRows, flash};
use crate::Op;
use crate::load::Load;
use crate::tensor::live::Live;
use crate::tensor::memory::residency::Residency;
use crate::tensor::scalar::Value;
use core::{fmt, marker::PhantomData};
use flash::{Duo, Head, Span};
use furiosa_opt_std::prelude::*;
use std::{io, ops::Range};

type Rows<R, W, C> = HbmTensor<bf16, C, m![R, W]>;
type Source<'a, R, W, C> = Live<&'a Rows<R, W, C>, R>;
type ZeroSpan<C> = DmTensor<bf16, C, m![Duo], m![Span / 8], m![Span % 8, Head]>;

#[derive(Clone, Copy, Default)]
struct Placement {
    source: usize,
    destination: usize,
    rows: usize,
}

/// A valid-only gather from initialized rows of `R` into the prefix of `Spans` key spans.
///
/// Construct on the host, upload once, then decode before numerical work. All counts and
/// offsets use individual rows, never projection blocks or modality strips. Source reads may
/// overlap; destination runs must cover a contiguous prefix exactly once. No modality or model
/// constants are owned here. Unused destination rows are initialized by [`PackKeys`].
pub struct PackPlan<R, Spans, const N: usize> {
    start: usize,
    rows: usize,
    count: usize,
    keys: usize,
    runs: [Placement; N],
    shape: PhantomData<(R, Spans)>,
}

impl<R, S, const N: usize> Copy for PackPlan<R, S, N> {}
impl<R, S, const N: usize> Clone for PackPlan<R, S, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: AxisName, S: AxisName, const N: usize> fmt::Debug for PackPlan<R, S, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PackPlan")
            .field("source", &(self.start..self.start + self.rows))
            .field("runs", &self.count)
            .field("keys", &self.keys)
            .field("source_capacity", &R::SIZE)
            .field("key_capacity", &S::SIZE.checked_mul(flash::Span::SIZE))
            .finish()
    }
}

impl<R: AxisName, S: AxisName, const N: usize> PackPlan<R, S, N> {
    /// Check every range, its transport representation, and complete destination coverage
    /// before uploading anything. Empty run lists are allowed; individual runs must be nonempty.
    pub fn try_from_runs(
        source: Range<usize>,
        runs: impl IntoIterator<Item = (Range<usize>, Range<usize>)>,
    ) -> io::Result<Self> {
        if source.start > source.end || source.end > R::SIZE {
            return Err(invalid(
                "initialized source interval is outside the allocation",
            ));
        }
        let capacity = S::SIZE
            .checked_mul(flash::Span::SIZE)
            .filter(|&capacity| capacity > 0)
            .ok_or_else(|| invalid("key span capacity must be positive and fit usize"))?;
        let mut plan = Self {
            start: source.start,
            rows: source.end - source.start,
            count: 0,
            keys: 0,
            runs: [Placement::default(); N],
            shape: PhantomData,
        };
        encoded(plan.start)?;
        encoded(plan.rows)?;
        for (from, to) in runs {
            if plan.count == N {
                return Err(invalid("too many valid runs"));
            }
            if from.start >= from.end || from.start < source.start || from.end > source.end {
                return Err(invalid(
                    "run must be nonempty and contained in initialized source rows",
                ));
            }
            let length = from.end - from.start;
            if to.start != plan.keys || to.end < to.start || to.end - to.start != length {
                return Err(invalid(
                    "destination runs must cover the key prefix contiguously at equal lengths",
                ));
            }
            if to.end > capacity {
                return Err(invalid("valid keys exceed the key allocation"));
            }
            encoded(from.start)?;
            encoded(to.start)?;
            encoded(length)?;
            encoded(to.end)?;
            plan.runs[plan.count] = Placement {
                source: from.start,
                destination: to.start,
                rows: length,
            };
            plan.count += 1;
            plan.keys = to.end;
        }
        encoded(plan.count)?;
        Ok(plan)
    }

    pub fn source_work(&self) -> Live<(), R> {
        Live::all().subrange(self.start, self.rows)
    }
    pub fn key_count(&self) -> usize {
        self.keys
    }
    pub fn run_count(&self) -> usize {
        self.count
    }

    /// Host inspection of the exact source/destination runs in their transfer order.
    pub fn runs(&self) -> impl Iterator<Item = (Range<usize>, Range<usize>)> + '_ {
        self.runs.iter().take(self.count).map(|run| {
            (
                run.source..run.source + run.rows,
                run.destination..run.destination + run.rows,
            )
        })
    }

    /// Number of valid-row TDMA operations per plane, excluding zero-span stores.
    pub fn copy_count(&self) -> usize {
        self.runs
            .iter()
            .take(self.count)
            .map(|run| {
                run.rows / flash::Span::SIZE + (run.rows % flash::Span::SIZE).count_ones() as usize
            })
            .sum()
    }

    /// `[source_start, source_rows, run_count, key_count]` and `[source_start, valid_rows]` per slot.
    /// Construction checked I32 representability. Unused slots are exactly zero.
    /// This inspection format is unchanged; the private upload also carries each checked
    /// destination start so device address expressions do not accumulate across runs.
    pub fn row_counts(&self) -> ([i32; 4], [[i32; 2]; N]) {
        (
            [
                self.start as i32,
                self.rows as i32,
                self.count as i32,
                self.keys as i32,
            ],
            self.runs.map(|run| [run.source as i32, run.rows as i32]),
        )
    }
}

#[derive(Clone, Debug)]
struct Slots<const N: usize>;
impl<const N: usize> AxisName for Slots<N> {
    const NAME: Ident = Ident::new("PackSlots");
    const SIZE: usize = N;
}

#[derive(DeviceSend)]
struct RunInput<C: M> {
    source: Value<C>,
    destination: Value<C>,
    rows: Value<C>,
}

/// Immutable request transport; fields are private and only a checked [`PackPlan`] can upload it.
/// Retain it until all launches borrowing it finish. Decode once at entry and share the local
/// plan between K and V; reading counts between transfers can wait for previous DMA work.
#[derive(DeviceSend)]
pub struct PackInputs<R, S, C: M, const N: usize> {
    start: Value<C>,
    rows: Value<C>,
    count: Value<C>,
    keys: Value<C>,
    runs: [RunInput<C>; N],
    shape: PhantomData<(R, S)>,
}

impl<R: AxisName, S: AxisName, C: M, const N: usize> PackInputs<R, S, C, N> {
    pub async fn upload(device: &mut Device, plan: &PackPlan<R, S, N>) -> io::Result<Self> {
        let (header, _) = plan.row_counts();
        let mut upload = async |value| {
            HostTensor::<i32, m![1]>::from_vec([value])
                .to_hbm::<C, m![1]>(&mut device.pdma)
                .await
                .map_err(io::Error::other)
        };
        let start = upload(header[0]).await?;
        let rows = upload(header[1]).await?;
        let count = upload(header[2]).await?;
        let keys = upload(header[3]).await?;
        let mut runs = Vec::with_capacity(N);
        for run in plan.runs {
            runs.push(RunInput {
                source: upload(run.source as i32).await?,
                destination: upload(run.destination as i32).await?,
                rows: upload(run.rows as i32).await?,
            });
        }
        Ok(Self {
            start,
            rows,
            count,
            keys,
            runs: runs
                .try_into()
                .unwrap_or_else(|_| unreachable!("one transport per checked slot")),
            shape: PhantomData,
        })
    }

    /// The same immutable row-count tensor may supply Compact's visible-key limit.
    pub fn key_count(&self) -> &Value<C> {
        &self.keys
    }

    /// Decode before numerical work. The opaque payload comes only from host validation;
    /// source-prefix containment and contiguous destination coverage are checked there.
    /// Individual fields are bounded here, and the leaf bounds each source/destination
    /// transfer to its allocation. Keeping direct scalar fields avoids recursively expanded
    /// address expressions on the retained compiler.
    pub fn read(&self) -> PackPlan<R, S, N> {
        const {
            assert!(N.checked_mul(R::SIZE).is_some());
        }
        let start = self.start.extent::<m![R]>();
        let rows = self.rows.extent::<m![R]>();
        let count = self.count.extent::<m![{ Symbol<Slots<N>> }]>();
        let keys = self.keys.extent::<m![{ Symbol<KeyRows<S>> }]>();
        let mut index = 0;
        let runs = [(); N].map(|_| {
            let source = self.runs[index].source.extent::<m![R]>();
            let destination = self.runs[index]
                .destination
                .extent::<m![{ Symbol<KeyRows<S>> }]>();
            let rows = self.runs[index].rows.extent::<m![R]>();
            index += 1;
            Placement {
                source,
                destination,
                rows,
            }
        });
        PackPlan {
            start,
            rows,
            count,
            keys,
            runs,
            shape: PhantomData,
        }
    }
}

/// Valid rows copied without arithmetic into head-major K/V, with a finite zero suffix.
/// Use the same decoded plan for rotated K and unrotated V. Feature pairing belongs to their
/// producers; this operation changes row/head layout only.
#[derive(DeviceSend, Load, Residency)]
pub struct PackKeys<S>(PhantomData<S>);
impl<S> Default for PackKeys<S> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<S: AxisName> PackKeys<S> {
    /// Initialize all destination keys. `input` must initialize the plan's source interval;
    /// its Live metadata must contain that interval. Each valid run must contain finite values.
    /// Rows outside valid runs may be NaN, including modality guards and the allocation tail.
    /// Complete trailing spans are cleared BEFORE copies, so the valid portion of a partial
    /// span is restored before return. No valid source run is rounded up or over-read.
    pub fn write<R: AxisName, W: AxisName, C: M, const N: usize>(
        &self,
        device: &mut Device,
        input: Source<'_, R, W, C>,
        plan: &PackPlan<R, S, N>,
        output: &mut flash::Tokens<S, flash::HeadPairs<W>, C>,
    ) {
        const {
            assert!(W::SIZE > 0 && W::SIZE.is_multiple_of(flash::Duo::SIZE * flash::Head::SIZE));
            assert!(flash::Span::SIZE == 2048);
        }
        self.clear::<W, C>(device, output, plan.keys / flash::Span::SIZE);
        // Aggregate iteration avoids a runtime select across the decoded scalar records.
        let _ = plan.runs.map(|run| {
            self.copy_run(
                device,
                input.value(),
                output,
                run.source,
                run.destination,
                run.rows,
            );
        });
    }

    // Fixed-length views retain the source and destination capacity strides. This is the same
    // mapping as static Keys, with whole spans and exact binary tails instead of padded runs.
    fn copy_run<R: AxisName, W: AxisName, C: M>(
        &self,
        device: &mut Device,
        input: &HbmTensor<bf16, C, m![R, W]>,
        output: &mut flash::Tokens<S, flash::HeadPairs<W>, C>,
        from: usize,
        to: usize,
        rows: usize,
    ) {
        let from = from.min(R::SIZE);
        let to = to.min(KeyRows::<S>::SIZE);
        let rows = rows.min(R::SIZE - from).min(KeyRows::<S>::SIZE - to);
        if const { R::SIZE >= 2048 && KeyRows::<S>::SIZE >= 2048 } {
            for piece in 0..rows / 2048 {
                CopyRows::<S, 2048>(PhantomData).forward(
                    device,
                    (
                        (input, from + piece * 2048),
                        (&mut *output, to + piece * 2048),
                    ),
                );
            }
        }
        // Higher bits give the offset, and this bit says whether its exact piece exists.
        // Derive each directly from `rows`; accumulating cursors recursively duplicates the
        // runtime address expressions in every subsequent fixed-length view.
        macro_rules! tail {
            ($length:literal) => {
                if const { R::SIZE >= $length && KeyRows::<S>::SIZE >= $length } {
                    let first = rows / (2 * $length) * (2 * $length);
                    for _ in 0..rows % (2 * $length) / $length {
                        CopyRows::<S, $length>(PhantomData)
                            .forward(device, ((input, from + first), (&mut *output, to + first)));
                    }
                }
            };
        }
        tail!(1024);
        tail!(512);
        tail!(256);
        tail!(128);
        tail!(64);
        tail!(32);
        tail!(16);
        tail!(8);
        tail!(4);
        tail!(2);
        tail!(1);
    }

    fn clear<W: AxisName, C: M>(
        &self,
        device: &mut Device,
        output: &mut flash::Tokens<S, flash::HeadPairs<W>, C>,
        first: usize,
    ) {
        type P<W> = Symbol<flash::HeadPairs<W>>;
        if first < S::SIZE {
            let mut zero: ZeroSpan<C> = DmTensor::new();
            zero.view_mut()
                .memset(const { bf16::from_f32(0.0) }, &mut device.sub);
            #[unroll]
            for pair in 0..P::<W>::SIZE {
                for offset in 0..S::SIZE - first {
                    let span = first + offset;
                    let target = output
                        .view_mut()
                        .tile::<m![{ P<W> }], 1, m![{ P<W> }=1#{!}({ P<W> }),Duo,S,Span,Head]>(pair)
                        .tile::<m![S], 1, m![{ P<W> }=1#{!}({ P<W> }),Duo,S=1#{!}S,Span,Head]>(
                            span,
                        );
                    zero.view().to_hbm_view(&mut device.tdma, target);
                }
            }
        }
    }
}

impl<'a, R: AxisName, W: AxisName, S: AxisName, C: M, const N: usize>
    Op<(
        Live<&'a HbmTensor<bf16, C, m![R, W]>, R>,
        &PackPlan<R, S, N>,
    )> for PackKeys<S>
{
    type Output = flash::Tokens<S, flash::HeadPairs<W>, C>;
    fn forward(
        &self,
        device: &mut Device,
        (input, plan): (
            Live<&'a HbmTensor<bf16, C, m![R, W]>, R>,
            &PackPlan<R, S, N>,
        ),
    ) -> Self::Output {
        let mut output = HbmTensor::new();
        self.write(device, input, plan, &mut output);
        output
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
fn encoded(value: usize) -> io::Result<i32> {
    i32::try_from(value).map_err(|_| invalid("row transport exceeds I32"))
}
