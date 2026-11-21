//! Geometry shared by request packing and build-time whole-transfer expansion.

pub const DEFAULT_MAX_TEXT: usize = 2048;
pub const PARTS: usize = 4;
pub const TEXT_STRIP: usize = 16;
pub const ROW_BLOCK: usize = 256;
pub const AUDIO_ROWS: usize = 112;
pub const VIDEO_ROWS: usize = 9_328;

/// Every aligned work extent between the smallest and largest valid request.
pub(crate) const TRANSFER_ROWS: [usize; TRANSFER_COUNT] = {
    let mut rows = [0; TRANSFER_COUNT];
    let mut step = 0;
    while step < TRANSFER_COUNT {
        rows[step] = FIRST_TRANSFER + step * ROW_BLOCK;
        step += 1;
    }
    rows
};

const FIRST_TRANSFER: usize =
    (AUDIO_ROWS + VIDEO_ROWS + TEXT_STRIP).div_ceil(ROW_BLOCK) * ROW_BLOCK;
const LAST_TRANSFER: usize = {
    assert!(DEFAULT_MAX_TEXT > 0);
    let text = DEFAULT_MAX_TEXT.div_ceil(PARTS * TEXT_STRIP) * TEXT_STRIP;
    (AUDIO_ROWS + VIDEO_ROWS + text).div_ceil(ROW_BLOCK) * ROW_BLOCK
};
const TRANSFER_COUNT: usize = (LAST_TRANSFER - FIRST_TRANSFER) / ROW_BLOCK + 1;
