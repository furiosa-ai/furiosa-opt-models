//! Mistral NeMo's static-scale FP8 checkpoint on two chips over E4M3 records, served by its own
//! serving step: one launch a call (a request under the fixed tiling), planned on the host.
//!
//! - A call's one-row requests are decode rows, the others one prompt chunk of a segment each,
//!   from the first wave past the decode rows; the planner picks each segment's attention spans
//!   and whether the chunk runs on the TP2 body or the hidden-sharded route.
//! - Each layer's cache pages are the call's pool: a request's pages are those its block table
//!   names, and padding rows write the layer's scratch pages, which no request names.
//! - The launch reads its weights from the load and keeps its rows' state on the device between
//!   calls; a call copies its token IDs in and its kept rows' greedy tokens out.

mod load;

use super::{Batch, CacheSpec, Decoder, Group, Kind, Limits, Output, Pool, Rows};
use crate::{Checkpoint, LoadError};
use furiosa_opt_models::models::decoder::attention::mistral;
use furiosa_opt_models::models::decoder::{Config as _, Form, form};
use furiosa_opt_models::models::{Class, Family, Implements};
use furiosa_opt_models::ops::attention::full::paged::Tiling;
use furiosa_opt_models::ops::index::copy;
use furiosa_opt_models::ops::rotary::PairFrequencies;
use furiosa_opt_models::tensor::memory::{Dtype, alias};
use furiosa_opt_models::tensor::parallel::{Parallel, Parallelism};
use furiosa_opt_nemo::models::nemo::{LAYERS, parameters, plan, serving};
use furiosa_opt_std::prelude::*;
use furiosa_opt_std::runtime::launch;
use serving::attention::{self, Blocks, Head, Heads, Kv, Page, Tokens};
use serving::qkv::rotation::THETA;
use serving::{mlp::Capacity, step};
use std::convert::Infallible;
use std::io;
use std::sync::{Mutex, PoisonError};

/// The chips the step runs on, as the engine's copies address them.
type Two = <Parallel<2> as Parallelism>::Chip;

/// Bytes a layer's page takes on each chip: `Page` tokens' records of every head's key and value.
const PAGE: usize = Page::SIZE * Heads::SIZE * Kv::SIZE * Head::SIZE;

/// Bytes of a selected row's record of the step's tokens, whose first word is its token.
const TOKEN: usize = 4 * size_of::<i32>();

/// The checkpoint's weights in the step's layouts and the step's state on the device.
pub struct Nemo {
    model: parameters::Model,
    resident: Mutex<Resident>,
}

/// The state every launch binds: its rows between pieces, its attention partials and its readout.
struct Resident {
    state: step::State,
    /// The storage of `state.tokens`, which a call's output copies read.
    tokens: HbmBuffer,
}

/// A checkpoint is NeMo's when its config is the dense NeMo decoder's, rotates its heads as the
/// step does, and declares the static-scale E4M3 projections the step reads.
impl Implements for Nemo {
    const SIZE: &'static str = "Nemo";
    const CLASS: Class = Class {
        family: "mistral",
        name: "ForCausalLM",
    };

    fn scope(source: &Checkpoint) -> Result<&'static str, LoadError> {
        let refusal = |error: io::Error| {
            let (name, reason) = ("config.json".into(), error.to_string());
            match error.kind() {
                io::ErrorKind::Unsupported => LoadError::Unsupported { name, reason },
                _ => LoadError::InvalidData { name, reason },
            }
        };
        let config = <mistral::Config>::parse(source.config()).map_err(refusal)?;
        // The step turns each head's pairs at the unscaled frequencies of base `THETA`.
        let unscaled: Vec<f64> = (0..Head::SIZE / 2)
            .map(|pair| f64::from(THETA).powf(-2.0 * pair as f64 / Head::SIZE as f64))
            .collect();
        if config.frequencies().map_err(refusal)?
            != PairFrequencies::new(&unscaled, 1.0).map_err(refusal)?
        {
            return Err(LoadError::Unsupported {
                name: "config.json:rope_parameters".into(),
                reason: format!("the step rotates by the unscaled spectrum of theta {THETA}"),
            });
        }
        <form::TensorScaled as Form>::REQUIREMENT
            .admits(source)
            .map(|()| "")
    }
}

/// A launch runs the step over its rows: its decode rows from row 0, its prompt chunk from the
/// next whole wave; every kept row reads out its greedy token. The adaptive tiling launches once a
/// call; the fixed tiling once a request, so no row's sums see another request's rows.
impl Decoder for Nemo {
    const OUTPUT: Output = Output::Tokens;
    const CHIPS: usize = 2;
    const KV_CACHE: Dtype = Dtype::F8_E4M3;
    type Output = Rows<i32, m![1]>;
    type Layers = attention::Pool;
    type Adapter = Infallible;

    fn limits(&self) -> Limits {
        Limits {
            window: Blocks::SIZE * Tokens::SIZE,
            rows: Capacity::SIZE,
            cache: CacheSpec {
                pools: vec![Pool {
                    block: Page::SIZE,
                    bytes: PAGE,
                }],
                groups: vec![Group {
                    pool: 0,
                    layers: 0..LAYERS,
                    kind: Kind::Paged,
                    window: None,
                    scratch: plan::SCRATCH_PAGES,
                }],
            },
            adapters: 0,
        }
    }

    fn layers(storage: &[HbmBuffer]) -> io::Result<attention::Pool> {
        (storage.iter())
            .flat_map(|layer| {
                (0..layer.size() / PAGE).map(move |page| {
                    (layer.alias(page * PAGE..(page + 1) * PAGE))
                        .and_then(HbmTensor::try_from)
                        .map_err(io::Error::other)
                })
            })
            .collect::<io::Result<Vec<_>>>()
            .map(Into::into)
    }

    fn call(
        &self,
        device: &mut Device,
        pool: &mut attention::Pool,
        batch: Batch<'_, Infallible>,
        tiling: Tiling,
    ) -> impl Future<Output = io::Result<Rows<i32, m![1]>>> + Send {
        async move {
            let refused = |error: plan::Error| io::Error::new(io::ErrorKind::InvalidInput, error);
            let appended = batch.appended(0);
            let launches = match tiling {
                Tiling::Adaptive => vec![appended],
                Tiling::Fixed => appended.chunks(1).collect(),
            };
            let kept = batch.kept().len();
            let output = HbmBuffer::alloc(device, kept.max(1) * size_of::<i32>())
                .map_err(io::Error::other)?;
            for requests in launches {
                // The launch's call rows.
                let first = requests.first().map_or(0, |request| request.rows.start);
                let end = requests.last().map_or(0, |request| request.rows.end);
                let decode = (requests.iter())
                    .take_while(|request| request.rows.len() == 1)
                    .count();
                let chunked = decode < requests.len();
                let start = plan::chunk_start(decode);
                // A call row's capacity row: the chunk starts at the wave after the decode rows.
                let capacity = |row: usize| {
                    let row = row - first;
                    if row < decode {
                        row
                    } else {
                        row - decode + start
                    }
                };
                let pages = plan::Pages::new(
                    pool.len() / LAYERS,
                    (requests.iter())
                        .flat_map(|request| request.blocks.iter().map(|&id| id as usize)),
                )
                .map_err(refused)?;
                let mut rows = Vec::with_capacity(Capacity::SIZE);
                let mut base = 0;
                for request in requests {
                    if rows.len() == decode && chunked {
                        rows.extend((decode..start).map(|slot| plan::Row::padding(&pages, slot)));
                    }
                    rows.extend((0..request.rows.len()).map(|row| plan::Row {
                        token: 0,
                        base,
                        position: request.cached + row,
                    }));
                    base += request.blocks.len();
                }
                // The launch's kept rows, each with its row of the call's output.
                let selected = (batch.kept().iter().enumerate())
                    .filter(|&(_, row)| (first..end).contains(row))
                    .collect::<Vec<_>>();
                let planned = plan::Launch {
                    rows,
                    decode,
                    start,
                    selected: selected.iter().map(|&(_, &row)| capacity(row)).collect(),
                };
                // Rows past the launch's embed token 0, a row of the table.
                let zeros = vec![0_u8; Two::SIZE * Capacity::SIZE * size_of::<i32>()];
                let ids = (HostBufferSlice::from(&zeros[..])
                    .to_hbm(&mut device.pdma)
                    .await)
                    .map_err(io::Error::other)?;
                let inputs =
                    (batch.tokens().iter()).filter(|(_, rows)| (first..end).contains(&rows.start));
                let mut targets = (inputs.clone())
                    .map(|(_, rows)| {
                        let at = capacity(rows.start) * size_of::<i32>();
                        ids.alias(at..at + rows.len() * size_of::<i32>())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(io::Error::other)?;
                let sources = inputs.map(|(tokens, _)| tokens.buffer());
                copy::<Two>(&mut *device, sources.zip(&mut targets))?;
                let ids = alias(&ids, ..)?;
                let input = (planned.input(device, &pages, ids).await).map_err(refused)?;
                let mut resident = self.resident.lock().unwrap_or_else(PoisonError::into_inner);
                let Resident { state, tokens } = &mut *resident;
                std::mem::swap(pool, &mut state.pool);
                let submitted = launch(
                    step::forward,
                    (&mut *device, &self.model, (&input, &mut *state)),
                )
                .submit()
                .accepted();
                std::mem::swap(pool, &mut state.pool);
                submitted.map(drop).map_err(io::Error::other)?;
                let mut targets = (selected.iter())
                    .map(|&(at, _)| {
                        output.alias(at * size_of::<i32>()..(at + 1) * size_of::<i32>())
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(io::Error::other)?;
                let sources = (0..selected.len())
                    .map(|row| tokens.alias(row * TOKEN..row * TOKEN + size_of::<i32>()))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(io::Error::other)?;
                copy::<Two>(&mut *device, sources.iter().zip(&mut targets))?;
            }
            let output = output
                .alias(..kept * size_of::<i32>())
                .map_err(io::Error::other)?;
            Rows::new(&output, 0..kept)
        }
    }
}
