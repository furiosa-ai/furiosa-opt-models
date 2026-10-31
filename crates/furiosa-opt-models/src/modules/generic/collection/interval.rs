use std::ops::Range;

use crate::load::LoadError;
use furiosa_opt_std::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, DeviceSend)]
pub struct Interval<T> {
    origin: T,
    pairs: T,
}

impl Interval<usize> {
    pub fn new(range: Range<usize>, depth: usize) -> Result<Self, LoadError> {
        if range.is_empty() || range.end > depth {
            return Err(LoadError::InvalidData {
                name: "layers".into(),
                reason: format!("{range:?} is not a nonempty interval within depth {depth}"),
            });
        }
        Ok(Self {
            origin: range.start,
            pairs: range.len() - 1,
        })
    }

    pub const fn origin(&self) -> usize {
        self.origin
    }

    pub const fn count(&self) -> usize {
        self.pairs + 1
    }

    pub fn range(&self) -> Range<usize> {
        self.origin..self.origin + self.count()
    }
}
