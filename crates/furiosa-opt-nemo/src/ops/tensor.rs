//! Typed tensors over device storage, and the storage dtype of a scalar type.

use furiosa_opt_std::prelude::{HbmTensor, M, Scalar};
use furiosa_opt_std::runtime::HbmBuffer as Buffer;
pub use safetensors::Dtype;
use std::io;

/// Return the storage dtype for supported device scalar types; refuse unsupported scalars.
pub fn dtype<D: Scalar>() -> io::Result<Dtype> {
    use std::any::TypeId;

    [
        (Dtype::BF16, TypeId::of::<furiosa_opt_std::prelude::bf16>()),
        (Dtype::F16, TypeId::of::<half::f16>()),
        (Dtype::F32, TypeId::of::<f32>()),
        (Dtype::I32, TypeId::of::<i32>()),
        (Dtype::I64, TypeId::of::<i64>()),
        (Dtype::U8, TypeId::of::<u8>()),
        (
            Dtype::F8_E4M3,
            TypeId::of::<furiosa_opt_std::prelude::f8e4m3>(),
        ),
    ]
    .into_iter()
    .find_map(|(dtype, scalar)| (scalar == TypeId::of::<D>()).then_some(dtype))
    .ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "unsupported tensor scalar"))
}

/// `range` of `buffer` as a tensor of its exact size: another owner of the same storage.
pub fn alias<D: Scalar, Chips: M, E: M>(
    buffer: &Buffer,
    range: impl std::ops::RangeBounds<usize>,
) -> io::Result<HbmTensor<D, Chips, E>> {
    HbmTensor::try_from(buffer.alias(range).map_err(io::Error::other)?).map_err(io::Error::other)
}

/// Split storage into consecutive `bytes`-sized blocks; refuse a partial or unaligned block.
pub fn blocks<T>(buffer: Buffer, bytes: usize) -> io::Result<Vec<T>>
where
    T: TryFrom<Buffer, Error = furiosa_opt_std::Error>,
{
    if bytes == 0
        || !bytes.is_multiple_of(8)
        || !buffer.addr().is_multiple_of(8)
        || !buffer.size().is_multiple_of(bytes)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "storage does not contain complete aligned blocks",
        ));
    }
    let count = buffer.size() / bytes;
    let (blocks, _) = (0..count).try_fold(
        (Vec::with_capacity(count), buffer),
        |(mut blocks, rest), _| {
            let (block, rest) = rest.split_at(bytes).map_err(io::Error::other)?;
            blocks.push(block.try_into().map_err(io::Error::other)?);
            io::Result::Ok((blocks, rest))
        },
    )?;
    Ok(blocks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::layout::axes::Eight;
    use furiosa_opt_std::prelude::{Broadcast, Device, Symbol, Topology, m};

    /// Storage splits only into whole, nonempty blocks that start 8-byte aligned.
    #[test]
    fn refuses_partial_or_unaligned_blocks() -> io::Result<()> {
        let device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
        let buffer = Buffer::alloc(&device, 72).map_err(io::Error::other)?;
        let split = |range: std::ops::Range<usize>, bytes| {
            let buffer = buffer.alias(range).map_err(io::Error::other)?;
            blocks::<HbmTensor<f32, m![1], m![Eight]>>(buffer, bytes)
        };
        assert_eq!(split(0..64, 32)?.len(), 2);
        for (range, bytes) in [(0..72, 32), (4..36, 32), (0..64, 0)] {
            assert_eq!(
                split(range, bytes).unwrap_err().kind(),
                io::ErrorKind::InvalidInput
            );
        }
        Ok(())
    }
}
