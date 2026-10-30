//! Leaf rules of FP8 that a load quantizes from BF16: each reads the BF16 weight a projection
//! stores and builds the host values a channel-scaled projection loads.

pub(crate) mod fp8;
