//! Each layer's position in a fixed interleaved order, computed at compile time.

pub(crate) const trait Entry: Copy {
    fn branch(self) -> usize;

    fn cache(self) -> Option<usize> {
        None
    }
}

/// Each layer's position in an interleaved order, per collection, and the order's period.
pub(crate) struct Schedule<const A: usize, const B: usize, const C: usize = 0> {
    pub positions: ([usize; A], [usize; B], [usize; C]),
    pub caches: ([usize; A], [usize; B]),
    pub period: usize,
    pub counts: [usize; 3],
}

impl<const A: usize, const B: usize, const C: usize> Schedule<A, B, C> {
    pub const fn new<T: [const] Entry>(order: &[T], caches: usize) -> Self {
        assert!(order.len() == A + B + C);
        let mut schedule = Self {
            positions: ([0; A], [0; B], [0; C]),
            caches: ([0; A], [0; B]),
            period: 1,
            counts: [0; 3],
        };
        let mut counts = [0; 3];
        let mut index = 0;
        while index < order.len() {
            let branch = order[index].branch();
            let position = counts[branch];
            match branch {
                0 => schedule.positions.0[position] = index,
                1 => schedule.positions.1[position] = index,
                2 => schedule.positions.2[position] = index,
                _ => panic!("schedule branch exceeds its collection count"),
            }
            if let Some(cache) = order[index].cache() {
                assert!(cache < caches);
                match branch {
                    0 => schedule.caches.0[position] = cache,
                    1 => schedule.caches.1[position] = cache,
                    _ => panic!("cache access needs two collections"),
                }
            }
            counts[branch] += 1;
            index += 1;
        }
        assert!(counts[0] == A && counts[1] == B && counts[2] == C);
        while schedule.period < order.len() {
            let mut equal = order.len().is_multiple_of(schedule.period);
            index = schedule.period;
            while index < order.len() {
                equal = equal && order[index].branch() == order[index % schedule.period].branch();
                index += 1;
            }
            if equal {
                break;
            }
            schedule.period += 1;
        }
        index = 0;
        while index < schedule.period && index < order.len() {
            schedule.counts[order[index].branch()] += 1;
            index += 1;
        }
        schedule
    }
}
