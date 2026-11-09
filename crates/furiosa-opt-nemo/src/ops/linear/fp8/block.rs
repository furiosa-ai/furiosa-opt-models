//! E4M3 projections whose activations quantize at run time in groups of 128 inputs and whose
//! weights carry one scale per 128 x 128 block.
//!
//! - A group's scale is its largest magnitude over the largest E4M3 value, never below `FLOOR`,
//!   so an all-zero group quantizes to zeros instead of dividing by zero.
//! - Each 128-input block contracts on its own; its FP32 sum takes the block's activation scale,
//!   then its weight scale, before the blocks add up.

use furiosa_opt_std::prelude::*;

// The inputs of one quantization group, apart from the group they belong to.
axes![Group = 128];

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    axes![
        Narrow = 768,
        Square = 768,
        Middle = 512,
        Keys = 256,
        Wide = 1024
    ];
}
