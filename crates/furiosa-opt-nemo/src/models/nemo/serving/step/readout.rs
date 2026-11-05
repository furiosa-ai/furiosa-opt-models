use super::*;
use crate::compose::{Tile as SelectionTile, Tiles};
use crate::ops::layout::Dm;

const _: () = assert!(Selected::SIZE % Rows::SIZE == 0 && Selected::SIZE <= 4 * Rows::SIZE);

impl Recipe<'_> {
    /// Reads the selected rows out: each live wave into a group-major buffer, then each 64-row
    /// group of the selected rows gathered into the final norm's layout, through the norm and
    /// head (two groups a head weight stream), then the greedy tokens of each group holding a selected row the sampler does not draw,
    /// then the sampled rows' drawn tokens.
    pub(super) fn finish(
        &self,
        device: &mut Device,
        (norm, input, waves): (&Norm, &Input, usize),
        state: &mut State,
    ) {
        // Every live wave with each 80-wide group's rows contiguous across waves: a wave copies as
        // 64 runs of 10,240 bytes, and a row is one position of the capacity axis.
        let mut grouped: Grouped = HbmTensor::new();
        for at in 0..waves {
            let rows = grouped.view_mut().tile::<m![Capacity], 64, m![
                Hidden / 80,
                Capacity = 64 #{!} Capacity,
                Hidden % 80
            ]>(at * 64);
            // SAFETY: the 64-row tile keeps the capacity stride; only its row axis is renamed.
            let rows = unsafe {
                rows.reshape::<Chips, m![Hidden / 80, Rows #{!} Capacity, Hidden % 80]>()
            };
            wave(&state.hidden, at).to_hbm_view(&mut device.tdma, rows);
        }
        // The selected rows' byte offsets into row-major rows of 10,240 bytes, as offsets of their
        // 160-byte group rows.
        let offsets: DmTensor<i32, Chips, m![1 # 2], m![1 # 256], m![Selected]> =
            input.selected.to_dm(&mut device.tdma);
        let positions: DmTensor<i32, Chips, m![1 # 2], m![1 # 256], m![Selected]> = device
            .main
            .begin(offsets.view())
            .fetch::<m![Selected / 8], m![Selected % 8]>()
            .collect::<m![Selected / 8], m![Selected % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::LogicRightShift, 6_i32)
            .vector_final()
            .commit_trim::<m![Selected % 8]>()
            .commit();
        let positions: HbmTensor<i32, Chips, m![Selected]> = positions.to_hbm(&mut device.tdma);
        let normalized = |device: &mut Device, group| {
            Self::normalized(device, (norm, &grouped, &positions), group)
        };
        let count = input.count.extent::<m![Selected]>();
        let groups = (count + 63) / 64;
        // The groups share one stream of the head weight, which bounds a group's head alone.
        let (four, three) = (groups / 4, groups / 3 * (1 - groups / 4));
        let (two, one) = (
            groups / 2 * (1 - groups / 3),
            (1 - groups / 2) * groups.min(1),
        );
        // A last group of at most 32 live rows contracts half its register: with three or four
        // groups the head's Main work outruns the weight stream.
        let whole_four = four * ((count + 223) / 224) - four;
        let whole_three = three * ((count + 159) / 160) - three;
        for _ in 0..whole_four {
            let a = normalized(device, 0);
            let b = normalized(device, 1);
            let c = normalized(device, 2);
            let d = normalized(device, 3);
            let (a, b, c, d) = self
                .head
                .apply_four::<head::Whole>(device, (&a, &b, &c, &d));
            store(device, &a, (&mut state.logits, 0));
            store(device, &b, (&mut state.logits, 1));
            store(device, &c, (&mut state.logits, 2));
            store(device, &d, (&mut state.logits, 3));
        }
        for _ in 0..four - whole_four {
            let a = normalized(device, 0);
            let b = normalized(device, 1);
            let c = normalized(device, 2);
            let d = normalized(device, 3);
            let (a, b, c, d) = self.head.apply_four::<head::Half>(device, (&a, &b, &c, &d));
            store(device, &a, (&mut state.logits, 0));
            store(device, &b, (&mut state.logits, 1));
            store(device, &c, (&mut state.logits, 2));
            store(device, &d, (&mut state.logits, 3));
        }
        for _ in 0..whole_three {
            let a = normalized(device, 0);
            let b = normalized(device, 1);
            let c = normalized(device, 2);
            let (a, b, c) = self.head.apply_three::<head::Whole>(device, (&a, &b, &c));
            store(device, &a, (&mut state.logits, 0));
            store(device, &b, (&mut state.logits, 1));
            store(device, &c, (&mut state.logits, 2));
        }
        for _ in 0..three - whole_three {
            let a = normalized(device, 0);
            let b = normalized(device, 1);
            let c = normalized(device, 2);
            let (a, b, c) = self.head.apply_three::<head::Half>(device, (&a, &b, &c));
            store(device, &a, (&mut state.logits, 0));
            store(device, &b, (&mut state.logits, 1));
            store(device, &c, (&mut state.logits, 2));
        }
        for _ in 0..two {
            let a = normalized(device, 0);
            let b = normalized(device, 1);
            let (a, b) = self.head.apply_pair(device, (&a, &b));
            store(device, &a, (&mut state.logits, 0));
            store(device, &b, (&mut state.logits, 1));
        }
        // A lone group of at most 32 live rows contracts half its register too: its last tile's
        // contraction, add and rounding are the head's exposed tail.
        let whole_one = one * ((count + 31) / 32) - one;
        for _ in 0..whole_one {
            let normalized = normalized(device, 0);
            let logits = self.head.apply::<head::Whole>(device, &normalized);
            store(device, &logits, (&mut state.logits, 0));
        }
        for _ in 0..one - whole_one {
            let normalized = normalized(device, 0);
            let logits = self.head.apply::<head::Half>(device, &normalized);
            store(device, &logits, (&mut state.logits, 0));
        }
        // The sampler draws the first `drawn` of the `count` selected rows (`drawn <= count`): greedy
        // tokens from group drawn / 64 on, and none of a partial last group whose every row it draws.
        let drawn = input.sampling.rows.extent::<m![Selected]>();
        // Each 1 or 0: the sampler draws every selected row; the last group is partial.
        let every = 1 - (count - drawn + 65535) / 65536;
        let partial = groups - count / 64;
        let greedy = groups - drawn / 64 - (every + partial) / 2;
        let first = drawn / 64;
        // The greedy groups share each chip-link crossing, whose latency bounds a group alone.
        let (four, three) = (greedy / 4, greedy / 3 * (1 - greedy / 4));
        let (two, one) = (
            greedy / 2 * (1 - greedy / 3),
            (1 - greedy / 2) * greedy.min(1),
        );
        for _ in 0..four {
            let a = scores(device, &state.logits, first);
            let b = scores(device, &state.logits, first + 1);
            let c = scores(device, &state.logits, first + 2);
            let d = scores(device, &state.logits, first + 3);
            let [a, b, c, d] = self.greedy.forward((
                &mut *device,
                Dm {
                    value: [&a, &b, &c, &d],
                    tiles: Tiles::<SelectionTile<Rows, Rows>>::over(Rows::SIZE),
                },
            ));
            publish(device, &a, (&mut state.tokens, first));
            publish(device, &b, (&mut state.tokens, first + 1));
            publish(device, &c, (&mut state.tokens, first + 2));
            publish(device, &d, (&mut state.tokens, first + 3));
        }
        for _ in 0..three {
            let a = scores(device, &state.logits, first);
            let b = scores(device, &state.logits, first + 1);
            let c = scores(device, &state.logits, first + 2);
            let [a, b, c] = self.greedy.forward((
                &mut *device,
                Dm {
                    value: [&a, &b, &c],
                    tiles: Tiles::<SelectionTile<Rows, Rows>>::over(Rows::SIZE),
                },
            ));
            publish(device, &a, (&mut state.tokens, first));
            publish(device, &b, (&mut state.tokens, first + 1));
            publish(device, &c, (&mut state.tokens, first + 2));
        }
        for _ in 0..two {
            let a = scores(device, &state.logits, first);
            let b = scores(device, &state.logits, first + 1);
            let [a, b] = self.greedy.forward((
                &mut *device,
                Dm {
                    value: [&a, &b],
                    tiles: Tiles::<SelectionTile<Rows, Rows>>::over(Rows::SIZE),
                },
            ));
            publish(device, &a, (&mut state.tokens, first));
            publish(device, &b, (&mut state.tokens, first + 1));
        }
        for _ in 0..one {
            let scores = scores(device, &state.logits, first);
            let tokens = self.greedy.forward((
                &mut *device,
                Dm {
                    value: &scores,
                    tiles: Tiles::<SelectionTile<Rows, Rows>>::over(Rows::SIZE),
                },
            ));
            publish(device, &tokens, (&mut state.tokens, first));
        }
        // The sampled rows' tokens replace their greedy ones.
        sample::draw(
            device,
            (&state.logits, &input.sampling),
            (&self.greedy.indices, &self.sampler),
            &mut state.tokens,
        );
    }

    /// Selected-row group `group` gathered into the final norm's layout, normalized and landed as
    /// the head's input.
    fn normalized(
        device: &mut Device,
        (norm, grouped, positions): (&Norm, &Grouped, &HbmTensor<i32, Chips, m![Selected]>),
        group: usize,
    ) -> head::Input {
        let offsets = positions
            .view()
            .tile::<m![Selected], 64, m![Selected = 64 # Selected]>(64 * group);
        // SAFETY: the offset tile keeps its stride and the group's row order.
        let offsets = unsafe { offsets.reshape::<Chips, m![Rows # Selected]>() };
        let mut indices: HbmTensor<i32, Chips, m![Rows]> = HbmTensor::new();
        offsets.to_hbm_view(&mut device.tdma, indices.view_mut());
        // The group's rows land in the norm's layout, four rows of a 160-wide group a slice.
        let selected: norm::Values = grouped
            .gather::<m![Capacity], m![Hidden / 80, Hidden % 80]>()
            .by_byte_offsets(&indices)
            .to_dm(&mut device.tdma);
        let normalized: norm::Input = norm
            .forward((&mut *device, &selected))
            .to_hbm(&mut device.tdma);
        normalized.to_dm(&mut device.tdma)
    }
}

type Grouped = HbmTensor<bf16, Chips, m![Hidden / 80, Capacity, Hidden % 80]>;

/// Group `group`'s logits into its 64 rows of `all`.
fn store(
    device: &mut Device,
    logits: &head::Logits,
    (all, group): (
        &mut HbmTensor<bf16, Chips, m![Selected, ShardVocabulary]>,
        usize,
    ),
) {
    let rows = all
        .view_mut()
        .tile::<m![Selected], 64, m![Selected = 64 #{!} Selected, ShardVocabulary]>(64 * group);
    // SAFETY: the 64-row tile keeps the selected stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chips, m![Rows #{!} Selected, ShardVocabulary]>() };
    logits.view().to_hbm_view(&mut device.tdma, rows);
}

/// Group `group`'s logits of `all`, landed for the greedy selection.
fn scores(
    device: &mut Device,
    all: &HbmTensor<bf16, Chips, m![Selected, ShardVocabulary]>,
    group: usize,
) -> head::greedy::Scores {
    let scores = all
        .view()
        .tile::<m![Selected], 64, m![Selected = 64 # Selected, ShardVocabulary]>(64 * group);
    // SAFETY: the 64-row tile keeps the selected stride; only its row axis is renamed.
    unsafe { scores.reshape::<Chips, m![Rows # Selected, ShardVocabulary]>() }
        .to_dm(&mut device.tdma)
}

/// Group `group`'s tokens into its 64 rows of `all`.
fn publish(
    device: &mut Device,
    tokens: &head::greedy::Tokens,
    (all, group): (&mut HbmTensor<i32, Chips, m![Selected, 1 # 4]>, usize),
) {
    let rows = all
        .view_mut()
        .tile::<m![Selected], 64, m![Selected = 64 #{!} Selected, 1 # 4]>(64 * group);
    // SAFETY: the 64-row tile keeps the selected stride; only its row axis is renamed.
    let rows = unsafe { rows.reshape::<Chips, m![Rows #{!} Selected, 1 # 4]>() };
    tokens.view().to_hbm_view(&mut device.tdma, rows);
}
