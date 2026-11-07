//! A rotated loop: the units of a periodic sequence, one trip holding the rest of a period and the
//! head of the next.

/// How many periods a [`window`] runs, and so its trips: one fewer.
pub trait Periods {
    fn trips(self) -> usize;
}

/// A runtime count of periods, 1 to `most`.
pub struct Bounded {
    pub periods: usize,
    pub most: usize,
}

impl Periods for Bounded {
    /// `periods - 1` for `periods` in `1..=most`, without a runtime subtraction.
    fn trips(self) -> usize {
        (self.periods + self.most - 1) % self.most
    }
}

/// Exactly `N` periods: a loop of constant trips.
pub struct Fixed<const N: usize>;

impl<const N: usize> Periods for Fixed<N> {
    fn trips(self) -> usize {
        const { assert!(N > 0, "a window runs a period") };
        N - 1
    }
}

/// A rotated loop over `periods` periods of a periodic unit sequence: `prologue` runs the head of
/// period 0, trip `k` the rest of period `k` and the head of period `k + 1` in the caller's
/// order, `epilogue` the rest of the last period, whose index it receives. A loop body is one
/// piece of task code, so the loop holds one trip's units, `periods - 1` times.
pub fn window<D, I, T, O>(
    context: &mut D,
    (input, periods): (I, impl Periods),
    (prologue, mut trip, epilogue): (
        impl FnOnce(&mut D, I) -> T,
        impl FnMut(&mut D, T, usize) -> T,
        impl FnOnce(&mut D, T, usize) -> O,
    ),
) -> O {
    let mut state = prologue(context, input);
    let last = periods.trips();
    for period in 0..last {
        state = trip(context, state, period);
    }
    epilogue(context, state, last)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The window runs the periodic sequence `h0 t0 h1 t1 …` rotated by its head: one trip a
    /// period after the first, up to its most periods or exactly its fixed ones.
    #[test]
    fn window_rotates_by_its_head() {
        fn run(periods: impl Periods) -> Vec<String> {
            window(
                &mut (),
                (Vec::new(), periods),
                (
                    |_, mut units: Vec<String>| {
                        units.push("h0".into());
                        units
                    },
                    |_, mut units, period| {
                        units.push(format!("t{period}"));
                        units.push(format!("h{}", period + 1));
                        units
                    },
                    |_, mut units, last| {
                        units.push(format!("t{last}"));
                        units
                    },
                ),
            )
        }
        let count = |periods| Bounded { periods, most: 3 };
        assert_eq!(run(count(1)), ["h0", "t0"]);
        assert_eq!(run(count(2)), ["h0", "t0", "h1", "t1"]);
        assert_eq!(run(count(3)), ["h0", "t0", "h1", "t1", "h2", "t2"]);
        assert_eq!(run(Fixed::<2>), ["h0", "t0", "h1", "t1"]);
    }
}
