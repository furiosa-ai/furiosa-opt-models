//! Each selected row's largest logits in every class of its vocabulary shard, from which the host
//! draws a sampled row exactly: the head's vocabulary order spreads a row's top tokens over the
//! classes, so a class rarely holds more of them than it returns.
//!
//! - A group is 1,024 consecutive entries of a row's shard; its entry `8 * member + class` belongs
//!   to class `class`, so a class is 128 entries of the group, one a member.
//! - A candidate is an entry's order key: its BF16 logit's signed monotone 16-bit key, then
//!   `127 - member` so a smaller member wins a tie, `(key << 7) | (127 - member)`, minus one.

use super::*;

axes![Candidates = 10, Member = 128, Class = 8];

type Cluster = m![ShardVocabulary / 32768];
/// A slice holds eight rows of one group.
type Slice = m![ShardVocabulary % 32768 / 1024, Rows / 8];
type Element = m![Rows % 8, Member, Class];
type Scores = DmTensor<bf16, Chips, Cluster, Slice, Element>;
type Keys = DmTensor<i32, Chips, Cluster, Slice, Element>;
type Found = DmTensor<i32, Chips, Cluster, Slice, m![Candidates, Rows % 8, Class]>;
/// A launch's selected rows' logits, as the readout leaves them.
type Logits = HbmTensor<bf16, Chips, m![Selected, ShardVocabulary]>;
/// The members' tie order, `127 - member`, in every slice.
pub type Members = HbmTensor<i32, Chips, m![{ Cluster }, { Slice }, Member, Class]>;
/// Each 64-row group's candidates of every class, largest first, as one slice finds them.
pub type Output = HbmTensor<
    i32,
    Chips,
    m![
        Selected / 64,
        { Cluster },
        { Slice },
        Candidates,
        Rows % 8,
        Class
    ],
>;

const _: () = assert!(Member::SIZE * Class::SIZE == 1024);

/// Every class's `Candidates` largest keys of the first `count` selected rows of `logits`.
#[device(chip = 2)]
pub fn select(
    device: &mut Device,
    (logits, count, members): (&Logits, &mlp::Count, &Members),
    output: &mut Output,
) {
    let count = count.extent::<m![Selected]>();
    for group in 0..(count + 63) / 64 {
        let rows = logits
            .view()
            .tile::<m![Selected], 64, m![Selected = 64 # Selected, ShardVocabulary]>(64 * group);
        // SAFETY: the 64-row tile keeps the selected stride; a shard entry is
        // `1024 * group + 8 * member + class`, so the axes only rename it.
        let rows = unsafe {
            rows.reshape::<Chips, m![Rows # Selected, ShardVocabulary / 1024, Member, Class]>()
        };
        let scores: Scores = rows.to_dm(&mut device.tdma);
        let order: DmTensor<i32, Chips, Cluster, Slice, m![Member, Class]> =
            members.to_dm(&mut device.tdma);
        let order = device
            .sub
            .begin(order.view())
            .fetch::<m![Member], m![Class]>()
            .collect::<m![Member], m![Class]>()
            .to_vrf::<m![Member, Class]>();
        let mut keys: Keys = device
            .main
            .begin(scores.view())
            .fetch::<m![Rows % 8, Member / 2], m![Member % 2, Class]>()
            .fetch_cast::<f32>()
            .collect::<m![Rows % 8, Member], m![Class]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_reinterpret::<i32>()
            .vector_stash()
            .vector_logic(LogicBinaryOpI32::ArithRightShift, 31i32)
            .vector_logic(LogicBinaryOpI32::BitAnd, i32::MAX)
            .vector_logic(LogicBinaryOpI32::BitXor, Stash)
            .vector_fxp(FxpBinaryOp::ArithRightShift, 16i32)
            .vector_fxp(FxpBinaryOp::LeftShift, 7i32)
            .vector_fxp(FxpBinaryOp::AddFxp, &order)
            .vector_final()
            .commit_trim::<m![Class]>()
            .commit();
        let mut found = Found::new();
        largest(device, &keys, &mut found, 0);
        #[unroll]
        for at in 1..10 {
            let best = found
                .view()
                .tile::<m![Candidates], 1, m![Candidates = 1 # 10, Rows % 8, Class]>(at - 1);
            let best = device
                .sub
                .begin(best)
                .fetch::<m![Rows % 8], m![Class]>()
                .collect::<m![Rows % 8], m![Class]>()
                .to_vrf::<m![Rows % 8, Class]>();
            // The candidate just found falls below every live key; the rest keep theirs.
            keys = device
                .main
                .begin(keys.view())
                .fetch::<m![Rows % 8, Member], m![Class]>()
                .collect::<m![Rows % 8, Member], m![Class]>()
                .vector_init()
                .vector_intra_slice_tag(TagMode::Zero)
                .vector_stash()
                .vector_fxp_with_mode(FxpBinaryOp::SubFxpSat, BinaryArgMode::Mode10, &best)
                .vector_fxp(FxpBinaryOp::ArithRightShift, 31i32)
                .vector_fxp(FxpBinaryOp::LeftShift, 24i32)
                .vector_clip(ClipBinaryOpI32::AddFxp, Stash)
                .vector_final()
                .commit_trim::<m![Class]>()
                .commit();
            largest(device, &keys, &mut found, at);
        }
        let slab = output.view_mut().tile::<m![Selected / 64], 1, m![
            Selected / 64 = 1 #{!} 4,
            { Cluster },
            { Slice },
            Candidates,
            Rows % 8,
            Class
        ]>(group);
        found.view().to_hbm_view(&mut device.tdma, slab);
    }
}

/// Candidate `at` of every class: the largest live key, minus one.
fn largest(device: &mut Device, keys: &Keys, found: &mut Found, at: usize) {
    let slot = found
        .view_mut()
        .tile::<m![Candidates], 1, m![Candidates = 1 #{!} 10, Rows % 8, Class]>(at);
    device
        .main
        .begin(keys.view())
        .fetch::<m![Rows % 8, Member], m![Class]>()
        .collect::<m![Rows % 8, Member], m![Class]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Rows % 8, Member, Class / 4], m![Class % 4]>()
        .vector_intra_slice_reduce::<Member, m![Rows % 8, Class / 4], m![Class % 4]>(
            IntraSliceReduceOpI32::Max,
        )
        .vector_widen_concat::<m![Rows % 8], m![Class]>()
        .vector_clip(ClipBinaryOpI32::AddFxp, -1i32)
        .vector_final()
        .commit_trim::<m![Class]>()
        .commit_view(slot);
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::runtime::launch;

    /// The monotone order key of a BF16 value, shifted under the member, as `select` builds it.
    fn order(value: bf16) -> i64 {
        let bits = (u32::from(value.to_bits()) << 16) as i32;
        i64::from((bits ^ ((bits >> 31) & i32::MAX)) >> 16) << 7
    }

    /// Every class keeps its 10 largest keys, ties to the smaller member, of each selected row
    /// below `count`, on both chips: what the host draws a row from.
    #[tokio::test(flavor = "current_thread")]
    async fn returns_every_class_its_largest_keys() {
        // Given
        let (shard, count) = (ShardVocabulary::SIZE, 70);
        let value = |chip: usize, row: usize, entry: usize| {
            let mixed = (chip * 7919 + row * 104_729 + entry * 31) % 977;
            match mixed % 11 {
                0 => bf16::from_f32(-0.0),
                1 => bf16::from_f32(f32::NEG_INFINITY),
                _ => bf16::from_f32((mixed % 64) as f32 * 0.25 - 8.0),
            }
        };
        let values: Vec<bf16> = (0..Chips::SIZE)
            .flat_map(|chip| (0..Selected::SIZE).map(move |row| (chip, row)))
            .flat_map(|(chip, row)| (0..shard).map(move |entry| value(chip, row, entry)))
            .collect();
        let mut device = Device::new(select.topology()).unwrap();
        let logits = HostTensor::<bf16, m![{ Chips }, Selected, ShardVocabulary]>::from_vec(values)
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let rows = HostTensor::<i32, m![{ Chips }, 1]>::from_vec([count as i32; 2])
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let members =
            HostTensor::<i32, m![{ Chips }, { Cluster }, { Slice }, Member, Class]>::from_vec(
                (0..Chips::SIZE * 512 * 1024).map(|at| 127 - i32::try_from(at % 1024 / 8).unwrap()),
            )
            .to_hbm(&mut device.pdma)
            .await
            .unwrap();
        let mut output = HostTensor::<
            i32,
            m![
                { Chips },
                Selected / 64,
                { Cluster },
                { Slice },
                Candidates,
                Rows % 8,
                Class
            ],
        >::zero()
        .to_hbm(&mut device.pdma)
        .await
        .unwrap();

        // When
        launch(
            select,
            (&mut device, (&logits, &rows, &members), &mut output),
        )
        .await
        .unwrap();

        // Then
        let actual = output
            .to_host::<m![
                { Chips },
                Selected / 64,
                { Cluster },
                { Slice },
                Candidates,
                Rows % 8,
                Class
            ]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec();
        for chip in 0..Chips::SIZE {
            for row in 0..count {
                for group in 0..64 {
                    for class in 0..8 {
                        let mut keys: Vec<i64> = (0..128)
                            .map(|member| {
                                let entry = 1024 * group + 8 * member + class;
                                order(value(chip, row, entry)) + 127 - member as i64
                            })
                            .collect();
                        keys.sort_unstable_by(|a, b| b.cmp(a));
                        for (at, key) in keys.iter().take(Candidates::SIZE).enumerate() {
                            let slice = (group % 32) * 8 + row % 64 / 8;
                            let index =
                                (((((chip * 4 + row / 64) * 2 + group / 32) * 256 + slice) * 10
                                    + at)
                                    * 8
                                    + row % 8)
                                    * 8
                                    + class;
                            assert_eq!(
                                i64::from(actual[index]),
                                key - 1,
                                "chip={chip} row={row} group={group} class={class} at={at}"
                            );
                        }
                    }
                }
            }
        }
    }
}
