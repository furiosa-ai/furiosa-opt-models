use std::ops::Range;

use crate::load::{Context, Load, LoadError};
use crate::tensor::scalar::Value;
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

pub trait Extent<const N: usize>: Sized {
    fn origin(&self) -> usize;
    fn count(&self) -> usize;
}

impl<const N: usize> Extent<N> for Interval<usize> {
    fn origin(&self) -> usize {
        self.origin()
    }
    fn count(&self) -> usize {
        self.count()
    }
}

#[derive(Debug, Clone)]
struct Pairs<const N: usize>;

impl<const N: usize> AxisName for Pairs<N> {
    const NAME: Ident = Ident::new("LayerPairs");
    const SIZE: usize = N - 1;
}

impl Load<&Interval<usize>> for Interval<usize> {
    fn load(_: &mut Context<'_>, from: &Self) -> Result<Self, LoadError> {
        Ok(*from)
    }
}

impl Load<&Interval<usize>> for Interval<Value> {
    fn load(context: &mut Context<'_>, interval: &Interval<usize>) -> Result<Self, LoadError> {
        let mut upload = |value: usize| {
            let value = i32::try_from(value).map_err(|error| LoadError::InvalidData {
                name: "layers".into(),
                reason: error.to_string(),
            })?;
            Value::load(context, HostTensor::<i32, m![1]>::from_vec([value]))
        };
        Ok(Self {
            origin: upload(interval.origin)?,
            pairs: upload(interval.pairs)?,
        })
    }
}

impl<'a, T> Load<&'a Interval<T>> for Interval<&'a T> {
    fn load(_: &mut Context<'_>, from: &'a Interval<T>) -> Result<Self, LoadError> {
        Ok(Self {
            origin: &from.origin,
            pairs: &from.pairs,
        })
    }
}

impl<const N: usize> Extent<N> for Interval<Value> {
    fn origin(&self) -> usize {
        self.origin.extent::<m![{ Symbol<super::Depth<N>> }]>()
    }

    fn count(&self) -> usize {
        self.pairs.extent::<m![{ Symbol<Pairs<N>> }]>() + 1
    }
}
