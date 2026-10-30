//! Rows selected, stored or copied by index.

pub use copy::copy;

mod copy;
pub mod gather;
pub mod packed;
pub mod scatter;
pub mod sequence;
pub mod table;
