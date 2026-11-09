//! Block-scaled E4M3 projections of resident rows: a launch's rows share each weight byte it loads.
//!
//! - A panel is a run of whole 128-input blocks; the weights are stored panel by panel. Each slice
//!   holds one panel of one owner's output rows, so it streams only its panel's weights.
//! - A cluster owns half the output rows; an owner, a quarter of a cluster's rows, takes one PE's
//!   slices (two PEs at long inputs). For example 5120 inputs make 40 panels of one block each.
//! - Each slice quantizes its panel of every row by the row's own 128-input group scale and holds
//!   the codes in its eight tensor lanes, one row a lane, as the weights stream past once.
//! - Each block's FP32 sum takes the row's activation scale, then the block's weight scale; the
//!   blocks then add up within the slice and the panels across the owner's slices.

use furiosa_opt_std::prelude::*;

// The tensor lanes the rows of a launch take, one row a lane.
axes![Lane = 8];
