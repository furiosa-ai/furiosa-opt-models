mod candidates;
pub mod sharded;
pub(crate) mod vector;

pub(super) use candidates::Candidates;
pub use vector::Elements;

/// Device-resident first-maximum token per row in the fused candidate reduction mapping.
pub type Selected<Rows> = candidates::Selected<Rows>;

/// Finite 2^30 mask penalty; padded logits must remain below valid logits after subtraction.
pub const PENALTY: f32 = 1_073_741_824.0;
