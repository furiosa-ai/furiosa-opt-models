use super::*;

pub(super) type Partial<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 16],
    m![
        Capacity % (Capacity / 4) / 8,
        Capacity / (Capacity / 4),
        Capacity % 8,
        Hidden % 16
    ],
>;
pub(super) type Half<Hidden, QWidth, Capacity> = DmTensor<
    bf16,
    Chip,
    Cluster<QWidth>,
    m![Hidden / 16],
    m![
        Capacity % 64 / 8,
        Capacity / (Capacity / 4),
        Capacity % 8,
        Hidden % 16
    ],
>;
