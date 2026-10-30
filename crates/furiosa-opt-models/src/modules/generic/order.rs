//! Each layer's position in a fixed interleaved order, computed at compile time.

use super::Branch;

/// Each layer's position in an interleaved order, per collection, and the order's period.
pub(crate) struct Order<const A: usize, const B: usize, const C: usize = 0> {
    pub positions: ([usize; A], [usize; B], [usize; C]),
    pub period: usize,
    pub counts: [usize; 3],
}

impl<const A: usize, const B: usize, const C: usize> Order<A, B, C> {
    pub(crate) const fn new(order: &[Branch]) -> Self {
        assert!(order.len() == A + B + C);
        let mut table = Self {
            positions: ([0; A], [0; B], [0; C]),
            period: 1,
            counts: [0; 3],
        };
        let mut counts = [0; 3];
        let mut index = 0;
        while index < order.len() {
            let branch = order[index] as usize;
            let position = counts[branch];
            match order[index] {
                Branch::First => table.positions.0[position] = index,
                Branch::Second => table.positions.1[position] = index,
                Branch::Third => table.positions.2[position] = index,
            }
            counts[branch] += 1;
            index += 1;
        }
        assert!(counts[0] == A && counts[1] == B && counts[2] == C);
        while table.period < order.len() {
            let mut equal = order.len().is_multiple_of(table.period);
            index = table.period;
            while index < order.len() {
                equal = equal && order[index] as usize == order[index % table.period] as usize;
                index += 1;
            }
            if equal {
                break;
            }
            table.period += 1;
        }
        index = 0;
        while index < table.period && index < order.len() {
            table.counts[order[index] as usize] += 1;
            index += 1;
        }
        table
    }
}
