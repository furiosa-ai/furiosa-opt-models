//! Compile-time contracts for rowwise media encoders whose outputs become language-model input rows.
//!
//! These capabilities describe positioned image-patch features and audio-frame samples. A model
//! with another input representation should get its own typed capability when it is added; these
//! contracts do not define a universal media container.

use crate::load::Load;
use furiosa_opt_std::prelude::{AxisName, DeviceSend};

/// One x/y coordinate of a positioned vision patch.
#[derive(Clone, Copy)]
pub enum Coordinate {
    /// Horizontal position.
    X,
    /// Vertical position.
    Y,
}

impl Coordinate {
    /// The coordinate's index in an `(x, y)` position.
    pub const fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Y => 1,
        }
    }
}

/// The common output row and loaded model view shared by an encoder's modalities.
///
/// Each capability computes one output row from each corresponding input row, in order, independent
/// of every other row. The executor leaves unused rows in a compiled launch uninitialized and
/// discards their outputs, so those rows must not affect valid outputs.
pub trait Encoder {
    /// A row of the model's output width.
    type Text: AxisName + Send + Sync;
    /// The loaded tree used to run this encoder.
    type View<'a>: Load<&'a Self>
    where
        Self: 'a;
}

/// A model capability that maps positioned image patches to output rows.
pub trait Vision: Encoder {
    /// The positive row count of the small vision launch variant.
    type SmallRows: AxisName + Send + Sync;
    /// The maximum row count of one large vision launch; no smaller than `SmallRows`.
    type LargeRows: AxisName + Send + Sync;
    /// A patch's feature width.
    type Patch: AxisName + Send + Sync;
    /// The extent of each coordinate in the model's position table.
    type Positions: AxisName + Send + Sync;
    /// State kept while one vision launch runs.
    type VisionState: Default + DeviceSend + Send;

    /// The byte offset for one coordinate of a patch position.
    fn offset(position: usize, coordinate: Coordinate) -> i32;
}

/// A model capability that maps audio-frame samples to output rows.
pub trait Audio: Encoder {
    /// The positive row count of the small audio launch variant.
    type SmallRows: AxisName + Send + Sync;
    /// The maximum row count of one large audio launch; no smaller than `SmallRows`.
    type LargeRows: AxisName + Send + Sync;
    /// A frame's sample width.
    type Samples: AxisName + Send + Sync;
    /// State kept while one audio launch runs.
    type AudioState: Default + DeviceSend + Send;
}
