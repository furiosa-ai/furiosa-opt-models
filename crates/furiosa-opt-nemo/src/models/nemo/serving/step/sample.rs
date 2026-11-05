//! The readout's device sampler over the selected rows, and its standalone check entries.

use super::*;
use head::sample::{self, Full};

type Logits = HbmTensor<bf16, Chips, m![Selected, ShardVocabulary]>;
/// One value a selected row, as the readout's tokens hold them.
pub type Rowwise<D> = HbmTensor<D, Chips, m![Selected, 1 # 4]>;
type Ids = HbmTensor<i32, Chips, m![ShardVocabulary]>;

/// A launch's sampled rows: the sampler draws selected rows `0..rows` in units of 16 (0 rows: no
/// sampler work), each row by its draw; a greedy row of a drawn batch draws (inverse 1, top-k 1,
/// top-p +inf, ban 0, uniform 0), which is its greedy token.
#[derive(DeviceSend)]
pub struct Plan {
    pub rows: mlp::Count,
    /// `1 / max(temperature, 1e-5)`.
    pub inverse: Rowwise<f32>,
    /// 1..=131072.
    pub top_k: Rowwise<i32>,
    /// +inf keeps the whole top-k.
    pub top_p: Rowwise<f32>,
    /// 1.0 bans the stop tokens.
    pub ban: Rowwise<f32>,
    /// In [0, 1).
    pub uniform: Rowwise<f32>,
}

/// The chain of 16-row units: each unit's searches run while the next unit lands.
mod units {
    use super::*;
    use head::sample::unit::{self as chain, Batch};

    /// A unit's HBM rows.
    type Unit<'a> =
        chain::Rows<'a, m![Batch # Selected, Full % 65536], m![Batch # Selected, 1 # 4]>;

    /// The HBM rows of the unit from selected row `start`.
    fn rows<'a>((logits, plan): (&'a Logits, &'a Plan), start: usize) -> Unit<'a> {
        let logits = logits
            .view()
            .tile::<m![Selected], 16, m![Selected = 16 # Selected, ShardVocabulary]>(start);
        // SAFETY: the tile keeps the selected stride; its rows are renamed to the unit's and its
        // entries to one half's.
        let logits = unsafe { logits.reshape::<Chips, m![Batch # Selected, Full % 65536]>() };
        chain::Rows {
            logits,
            inverse: parameter(&plan.inverse, start),
            top_k: parameter(&plan.top_k, start),
            top_p: parameter(&plan.top_p, start),
            ban: parameter(&plan.ban, start),
            uniform: parameter(&plan.uniform, start),
        }
    }

    fn parameter<D: VeScalar>(
        values: &Rowwise<D>,
        start: usize,
    ) -> HbmTensorView<'_, D, Chips, m![Batch # Selected, 1 # 4]> {
        let rows = values
            .view()
            .tile::<m![Selected], 16, m![Selected = 16 # Selected, 1 # 4]>(start);
        // SAFETY: the tile keeps the selected stride; only its row axis is renamed.
        unsafe { rows.reshape::<Chips, m![Batch # Selected, 1 # 4]>() }
    }

    fn store(
        device: &mut Device,
        value: &chain::Row<i32>,
        (output, start): (&mut Rowwise<i32>, usize),
    ) {
        let rows = output
            .view_mut()
            .tile::<m![Selected], 16, m![Selected = 16 #{!} Selected, 1 # 4]>(start);
        // SAFETY: the tile keeps the selected stride; only its row axis is renamed.
        let rows = unsafe { rows.reshape::<Chips, m![Batch #{!} Selected, 1 # 4]>() };
        chain::store(device, value, rows);
    }

    /// The `units` units from selected row 0 on (at least one): each row's token into `tokens`.
    /// The stop table and the token order land once: the order's masks while the first unit's peer
    /// half crosses the link, its ids under the first unit's searches. Every unit but the last lands
    /// the next one's rows under its searches.
    pub(super) fn draw(
        device: &mut Device,
        (logits, plan): (&Logits, &Plan),
        (ids, table): (&Ids, &sample::Table),
        (tokens, units): (&mut Rowwise<i32>, usize),
    ) {
        let stops = chain::stops(device, table);
        let mut ahead = chain::Ahead::new(device);
        ahead.land(device, rows((logits, plan), 0));
        ahead.exchange(device, &stops);
        let peer = ahead.send(device);
        let mut landed = chain::Landed::new(device);
        ahead.receive(device, &peer);
        // 1 when the launch draws one unit.
        for _ in 0..(16 - units) / 15 {
            let (searched, _) = chain::search(
                device,
                (&mut ahead, &stops),
                (None::<Unit>, Some((&mut landed, ids))),
            );
            let order = landed.order(device);
            let token = chain::finish(device, searched, &order);
            store(device, &token, (&mut *tokens, 0));
        }
        // 1 when it draws two units or more.
        for _ in 0..(units + 14) / 16 {
            let (searched, peer) = chain::search(
                device,
                (&mut ahead, &stops),
                (Some(rows((logits, plan), 16)), Some((&mut landed, ids))),
            );
            let order = landed.order(device);
            let token = chain::finish(device, searched, &order);
            if let Some(peer) = &peer {
                ahead.receive(device, peer);
            }
            store(device, &token, (&mut *tokens, 0));
            for at in 0..units - 2 {
                let next = rows((logits, plan), 16 * ((at + 2) % 16));
                let (searched, peer) =
                    chain::search(device, (&mut ahead, &stops), (Some(next), None));
                let token = chain::finish(device, searched, &order);
                if let Some(peer) = &peer {
                    ahead.receive(device, peer);
                }
                store(device, &token, (&mut *tokens, 16 * (at + 1)));
            }
            let (searched, _) = chain::search(device, (&mut ahead, &stops), (None::<Unit>, None));
            let token = chain::finish(device, searched, &order);
            store(device, &token, (&mut *tokens, 16 * ((units - 1) % 16)));
        }
    }
}

/// Every drawn row's token into `tokens`, from the selected rows' `logits`.
pub(super) fn draw(
    device: &mut Device,
    (logits, plan): (&Logits, &Plan),
    (ids, table): (&Ids, &sample::Table),
    tokens: &mut Rowwise<i32>,
) {
    let units = (plan.rows.extent::<m![Selected]>() + 15) / 16;
    // 1 when the launch draws rows (`units` is at most 16).
    for _ in 0..(units + 15) / 16 {
        units::draw(device, (logits, plan), (ids, table), (&mut *tokens, units));
    }
}

/// The sampler alone over the first `plan.rows` selected rows of `logits`.
#[device(chip = 2)]
pub fn sampled(
    device: &mut Device,
    (logits, plan, ids, table): (&Logits, &Plan, &Ids, &sample::Table),
    output: &mut Rowwise<i32>,
) {
    draw(device, (logits, plan), (ids, table), output);
}
