// The causal-history test every geometry's body passes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::nemo::serving::qkv::QHeads;
    use crate::models::nemo::serving::residual::Shard;
    use furiosa_opt_std::runtime::launch;

    const PAGE_BYTES: usize = Page::SIZE * Kv::SIZE * Heads::SIZE * Head::SIZE;

    /// The value of an E4M3 code, NaN codes aside.
    fn decode(code: u8) -> f64 {
        let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
        let (exponent, mantissa) = (i32::from(code >> 3 & 15), f64::from(code & 7) / 8.0);
        if exponent == 0 {
            sign * mantissa * 2f64.powi(-6)
        } else {
            sign * (1.0 + mantissa) * 2f64.powi(exponent - 7)
        }
    }

    /// A small deterministic E4M3 code: a sign, an exponent near one and three mantissa bits.
    fn code(seed: usize) -> u8 {
        let mut bits = (seed as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
        bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        bits ^= bits >> 31;
        (bits as u8 & 1) << 7 | (5 + (bits >> 4 & 3) as u8) << 3 | (bits >> 1 & 7) as u8
    }

    const START: usize = 64;

    /// The chunk from capacity row `START`, its span and block counts read from the plan as the
    /// step reads them, so each per-span and per-block guard stays a runtime choice.
    #[device(chip = 2)]
    fn chunk(
        device: &mut Device,
        (pool, plan, blocks, queries): (&Pool, &Tables, &PerSegment, &qkv::Queries),
        output: &mut Attended,
    ) {
        let mut landed = Landed::new();
        land(device, (pool, 0), &plan.pages[0], &mut landed);
        let mut sequence = Sequence::new(landed);
        let counts = (
            plan.spans.extent::<m![Spans]>(),
            plan.full.extent::<m![Spans]>(),
            plan.ahead.extent::<m![Reach]>(),
            blocks.extent::<m![QueryBlocks]>(),
        );
        sequence.attend_segment::<false>(
            device,
            (pool, 0),
            (plan, (0, START, 0), queries),
            (counts, 0),
            output,
        );
    }

    /// Each chunk row attends exactly the keys at and before its own position over every span its
    /// history reaches, unwritten NaN cells past the chunk and rows outside the chunk
    /// notwithstanding.
    #[tokio::test]
    async fn attends_each_row_over_its_causal_history() -> Result<(), Box<dyn std::error::Error>> {
        // Given: the chunk's rows straddle the first two spans; where a span holds several blocks,
        // only the second block's rows reach the second span.
        let lead = if Queries::SIZE < Keys::SIZE { Queries::SIZE } else { 0 };
        let (history, rows, start): (usize, usize, usize) =
            (Keys::SIZE - 9 - lead, 30 + lead, START);
        let length = history + rows;
        let spans = length.div_ceil(Keys::SIZE);
        let written = length.div_ceil(Page::SIZE);
        let mut context = Device::new(chunk.topology())?;
        let mut pool: Vec<HbmTensor<f8e4m3, Chip, m![Page, Heads, Kv, Head]>> = (0..written + 1)
            .map(|_| HbmTensor::alloc(&context))
            .collect::<Result<_, _>>()?;
        let cell = |chip: usize, token: usize, at: usize| -> u8 {
            if token < length {
                code((chip * 16384 + token) * 1024 + at)
            } else {
                0x7f
            }
        };
        for (page, block) in pool.iter_mut().enumerate() {
            // In the page's own order: the chips' host writes copy bytes, they do not transpose.
            let codes: Vec<u8> = (0..2 * PAGE_BYTES)
                .map(|at| {
                    let (chip, at) = (at / PAGE_BYTES, at % PAGE_BYTES);
                    let (token, head) = (at / 1024, at / 256 % 4);
                    let (kv, channel) = (at / 128 % 2, at % 128);
                    cell(chip, page * Page::SIZE + token, kv * 512 + head * 128 + channel)
                })
                .collect();
            HostTensor::<f8e4m3, m![Shard, Page, Heads, Kv, Head]>::from_buf(codes)
                .to_hbm(&mut context.pdma)
                .output(block)
                .await?;
        }
        let pages = pool.len();
        let pool: Pool = pool.into();
        let query = |chip: usize, row: usize, head: usize, channel: usize| -> f32 {
            (decode(code(((chip * 4096 + row) * 16 + head) * 128 + channel + 1)) / 4.0) as f32
        };
        let queries: qkv::Queries =
            HostTensor::<bf16, m![Shard, Capacity, QHeads, Head]>::from_vec(
                (0..2 * Capacity::SIZE * 16 * 128).map(|at| {
                    let (chip, row, head, channel) = (
                        at / (Capacity::SIZE * 2048),
                        at / 2048 % Capacity::SIZE,
                        at / 128 % 16,
                        at % 128,
                    );
                    bf16::from_f32(query(chip, row, head, channel))
                }),
            )
            .to_hbm(&mut context.pdma)
            .await?;
        let limits: Vec<f32> = (0..2 * Spans::SIZE * Prompt::SIZE * Lane::SIZE)
            .map(|at| {
                let (span, row) = (
                    at / (Prompt::SIZE * Lane::SIZE) % Spans::SIZE,
                    at / 8 % Prompt::SIZE,
                );
                let sees = (history + row.min(rows - 1)) as f32 - (span * Keys::SIZE) as f32;
                sees.clamp(-1.0, (Keys::SIZE - 1) as f32)
            })
            .collect();
        let mut tables = Vec::with_capacity(Spans::SIZE);
        for span in 0..Spans::SIZE {
            let offsets: Vec<i32> = (0..Keys::SIZE / Page::SIZE)
                .map(|page| {
                    i32::try_from((span * Keys::SIZE / Page::SIZE + page) % pages * PAGE_BYTES)
                })
                .collect::<Result<_, _>>()?;
            tables.push(
                HostTensor::<i32, m![Shard, Keys / 16]>::from_vec(
                    offsets.iter().chain(&offsets).copied(),
                )
                .to_hbm(&mut context.pdma)
                .await?,
            );
        }
        let counts = |value: usize| -> Result<HostTensor<i32, m![Shard, Segments]>, Box<dyn std::error::Error>> {
            Ok(HostTensor::from_vec(vec![i32::try_from(value)?; 2 * Segments::SIZE]))
        };
        let plan = Tables {
            spans: counts(spans)?.to_hbm(&mut context.pdma).await?,
            full: counts((history + 1) / Keys::SIZE)?
                .to_hbm(&mut context.pdma)
                .await?,
            ahead: counts(history.div_ceil(Queries::SIZE))?
                .to_hbm(&mut context.pdma)
                .await?,
            pages: tables.into(),
            limits: HostTensor::<f32, m![Shard, Spans, Prompt, Lane]>::from_vec(limits)
                .to_hbm(&mut context.pdma)
                .await?,
            positions: HostTensor::<f32, m![Shard, Keys]>::from_vec(
                (0..2 * Keys::SIZE).map(|at| (at % Keys::SIZE) as f32),
            )
            .to_hbm(&mut context.pdma)
            .await?,
        };
        let blocks = counts(rows.div_ceil(Queries::SIZE))?
            .to_hbm(&mut context.pdma)
            .await?;
        let mut output = Attended::alloc(&context)?;

        // When
        launch(chunk, (&mut context, (&pool, &plan, &blocks, &queries), &mut output)).await?;

        // Then
        let actual = output
            .to_host::<m![Shard, Prompt, Heads, Groups, Head]>(&mut context.pdma)
            .await?
            .into_vec();
        for chip in 0..2 {
            // Every cell decoded once: the reference reads each key and value for every row.
            let decoded: Vec<f64> = (0..length * 1024)
                .map(|at| decode(cell(chip, at / 1024, at % 1024)))
                .collect();
            let stored = |token: usize, at: usize| decoded[token * 1024 + at];
            // The rows by the block boundary: the first block's last rows end before span 1.
            for row in lead.saturating_sub(30)..rows {
                for head in 0..16 {
                    let kv = head / 4;
                    let q: Vec<f64> = (0..128)
                        .map(|channel| {
                            let query = query(chip, start + row, head, channel);
                            f64::from(bf16::from_f32(query).to_f32())
                        })
                        .collect();
                    let scores: Vec<f64> = (0..=history + row)
                        .map(|token| {
                            (0..128)
                                .map(|channel| q[channel] * stored(token, kv * 128 + channel))
                                .sum::<f64>()
                                / 128f64.sqrt()
                        })
                        .collect();
                    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> =
                        scores.iter().map(|score| (score - peak).exp()).collect();
                    let mass: f64 = weights.iter().sum();
                    let at = ((chip * Prompt::SIZE + row) * 16 + head) * 128;
                    let (error, energy) = (0..128).fold((0.0, 0.0), |(error, energy), channel| {
                        let expected = weights
                            .iter()
                            .enumerate()
                            .map(|(token, weight)| weight * stored(token, 512 + kv * 128 + channel))
                            .sum::<f64>()
                            / mass;
                        let actual = f64::from(actual[at + channel]);
                        (
                            error + (actual - expected).powi(2),
                            energy + expected.powi(2),
                        )
                    });
                    let relative = (error / energy).sqrt();
                    assert!(
                        relative < 1e-2,
                        "chip={chip} row={row} head={head} rel_l2={relative}"
                    );
                }
            }
        }
        Ok(())
    }

    /// The K cell a padding key holds under filler `filler`: 0 none, else magnitude code
    /// `filler` signed along the query channel of the row and query head the key aims at, so the
    /// key scores far above every visible key of that row.
    fn padding(filler: u8, chip: usize, token: usize, at: usize, rows: usize) -> u8 {
        let (kv, head, channel) = (at / 512, at / 128 % 4, at % 128);
        if filler == 0 || kv == 1 {
            return 0;
        }
        let (row, group) = (token % rows, token / rows % 4);
        let q = (decode(code(((chip * 4096 + START + row) * 16 + head * 4 + group) * 128 + channel + 1)) / 4.0) as f32;
        let sign = if bf16::from_f32(q).to_f32() < 0.0 { 0x80 } else { 0 };
        sign | filler
    }

    /// One fresh one-span chunk of `rows` rows after `history` keys, every key of the span from
    /// `length` on a padding key under `filler` (its value zero): the attended rows on the chips,
    /// and each row's largest relative L2 error against the FP64 softmax over the visible keys.
    async fn fresh_chunk(
        history: usize,
        rows: usize,
        filler: u8,
    ) -> Result<(Vec<f32>, Vec<f64>), Box<dyn std::error::Error>> {
        let length = history + rows;
        assert!(length <= Keys::SIZE);
        let pages = Keys::SIZE / Page::SIZE;
        let mut context = Device::new(chunk.topology())?;
        let mut pool: Vec<HbmTensor<f8e4m3, Chip, m![Page, Heads, Kv, Head]>> = (0..pages)
            .map(|_| HbmTensor::alloc(&context))
            .collect::<Result<_, _>>()?;
        let cell = |chip: usize, token: usize, at: usize| -> u8 {
            if token < length {
                code((chip * 16384 + token) * 1024 + at)
            } else {
                padding(filler, chip, token, at, rows)
            }
        };
        for (page, block) in pool.iter_mut().enumerate() {
            let codes: Vec<u8> = (0..2 * PAGE_BYTES)
                .map(|at| {
                    let (chip, at) = (at / PAGE_BYTES, at % PAGE_BYTES);
                    let (token, head) = (at / 1024, at / 256 % 4);
                    let (kv, channel) = (at / 128 % 2, at % 128);
                    cell(chip, page * Page::SIZE + token, kv * 512 + head * 128 + channel)
                })
                .collect();
            HostTensor::<f8e4m3, m![Shard, Page, Heads, Kv, Head]>::from_buf(codes)
                .to_hbm(&mut context.pdma)
                .output(block)
                .await?;
        }
        let pool: Pool = pool.into();
        let query = |chip: usize, row: usize, head: usize, channel: usize| -> f32 {
            (decode(code(((chip * 4096 + row) * 16 + head) * 128 + channel + 1)) / 4.0) as f32
        };
        let queries: qkv::Queries =
            HostTensor::<bf16, m![Shard, Capacity, QHeads, Head]>::from_vec(
                (0..2 * Capacity::SIZE * 16 * 128).map(|at| {
                    let (chip, row, head, channel) = (
                        at / (Capacity::SIZE * 2048),
                        at / 2048 % Capacity::SIZE,
                        at / 128 % 16,
                        at % 128,
                    );
                    bf16::from_f32(query(chip, row, head, channel))
                }),
            )
            .to_hbm(&mut context.pdma)
            .await?;
        let limits: Vec<f32> = (0..2 * Spans::SIZE * Prompt::SIZE * Lane::SIZE)
            .map(|at| {
                let (span, row) = (
                    at / (Prompt::SIZE * Lane::SIZE) % Spans::SIZE,
                    at / 8 % Prompt::SIZE,
                );
                let sees = (history + row.min(rows - 1)) as f32 - (span * Keys::SIZE) as f32;
                sees.clamp(-1.0, (Keys::SIZE - 1) as f32)
            })
            .collect();
        let mut tables = Vec::with_capacity(Spans::SIZE);
        for _ in 0..Spans::SIZE {
            let offsets: Vec<i32> = (0..pages)
                .map(|page| i32::try_from(page * PAGE_BYTES))
                .collect::<Result<_, _>>()?;
            tables.push(
                HostTensor::<i32, m![Shard, Keys / 16]>::from_vec(
                    offsets.iter().chain(&offsets).copied(),
                )
                .to_hbm(&mut context.pdma)
                .await?,
            );
        }
        let counts = |value: usize| -> Result<HostTensor<i32, m![Shard, Segments]>, Box<dyn std::error::Error>> {
            Ok(HostTensor::from_vec(vec![i32::try_from(value)?; 2 * Segments::SIZE]))
        };
        let plan = Tables {
            spans: counts(1)?.to_hbm(&mut context.pdma).await?,
            full: counts((history + 1) / Keys::SIZE)?
                .to_hbm(&mut context.pdma)
                .await?,
            ahead: counts(history.div_ceil(Queries::SIZE))?
                .to_hbm(&mut context.pdma)
                .await?,
            pages: tables.into(),
            limits: HostTensor::<f32, m![Shard, Spans, Prompt, Lane]>::from_vec(limits)
                .to_hbm(&mut context.pdma)
                .await?,
            positions: HostTensor::<f32, m![Shard, Keys]>::from_vec(
                (0..2 * Keys::SIZE).map(|at| (at % Keys::SIZE) as f32),
            )
            .to_hbm(&mut context.pdma)
            .await?,
        };
        let blocks = counts(rows.div_ceil(Queries::SIZE))?
            .to_hbm(&mut context.pdma)
            .await?;
        let mut output = Attended::alloc(&context)?;
        launch(chunk, (&mut context, (&pool, &plan, &blocks, &queries), &mut output)).await?;
        let actual = output
            .to_host::<m![Shard, Prompt, Heads, Groups, Head]>(&mut context.pdma)
            .await?
            .into_vec();
        let mut errors = vec![0f64; rows];
        for chip in 0..2 {
            let stored = |token: usize, at: usize| decode(cell(chip, token, at));
            for (row, worst) in errors.iter_mut().enumerate() {
                for head in 0..16 {
                    let kv = head / 4;
                    let q: Vec<f64> = (0..128)
                        .map(|channel| f64::from(bf16::from_f32(query(chip, START + row, head, channel)).to_f32()))
                        .collect();
                    let scores: Vec<f64> = (0..=history + row)
                        .map(|token| {
                            (0..128).map(|c| q[c] * stored(token, kv * 128 + c)).sum::<f64>()
                                / 128f64.sqrt()
                        })
                        .collect();
                    let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
                    let mass: f64 = weights.iter().sum();
                    let at = ((chip * Prompt::SIZE + row) * 16 + head) * 128;
                    let (error, energy) = (0..128).fold((0.0, 0.0), |(error, energy), channel| {
                        let expected = weights
                            .iter()
                            .enumerate()
                            .map(|(token, w)| w * stored(token, 512 + kv * 128 + channel))
                            .sum::<f64>()
                            / mass;
                        let actual = f64::from(actual[at + channel]);
                        (error + (actual - expected).powi(2), energy + expected.powi(2))
                    });
                    *worst = worst.max((error / energy).sqrt());
                }
            }
        }
        Ok((actual, errors))
    }

    /// A fresh chunk's attended rows do not depend on the keys past it, which no row sees:
    /// the same rows with zero padding keys and with padding keys scoring far above every visible
    /// key are equal, and both match the FP64 softmax over the visible keys.
    #[tokio::test]
    async fn padding_keys_leave_a_fresh_chunk_unmoved() -> Result<(), Box<dyn std::error::Error>> {
        let (history, rows) = (7, 30);
        // E4M3 codes 0x40 (2.0) and 0x7e (448): a sink-like key and an extreme one.
        let (zero, zero_errors) = fresh_chunk(history, rows, 0).await?;
        let mut failures = Vec::new();
        for filler in [0x40u8, 0x7e] {
            let (moved, errors) = fresh_chunk(history, rows, filler).await?;
            let (mut differing, mut largest) = (0usize, 0f32);
            for chip in 0..2 {
                for at in chip * Prompt::SIZE * 2048..(chip * Prompt::SIZE + rows) * 2048 {
                    if zero[at].to_bits() != moved[at].to_bits() {
                        differing += 1;
                        largest = largest.max((zero[at] - moved[at]).abs());
                    }
                }
            }
            let worst = errors.iter().copied().fold(0f64, f64::max);
            let zero_worst = zero_errors.iter().copied().fold(0f64, f64::max);
            println!(
                "COUNTERFACTUAL keys={} filler={filler:#04x} differing={differing}/{} max_abs={largest} rel_l2 zero={zero_worst:.3e} filler={worst:.3e}",
                Keys::SIZE,
                2 * rows * 2048
            );
            if differing != 0 || worst >= 1e-2 {
                failures.push(format!("filler={filler:#04x} differing={differing} rel_l2={worst}"));
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
        Ok(())
    }
}
