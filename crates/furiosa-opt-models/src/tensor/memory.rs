//! Where the leaves of a weight tree live, typed tensors over device storage, and the storage dtype
//! of a scalar type.
//!
//! - A tree is generic over its memory `X: Memory`; its leaves are `X::Tensor<D, C, S>` (scalar
//!   `D`, chip mapping `C`, element mapping `S`).
//! - [`Hbm`] keeps every leaf in device memory; `&X` borrows the leaves of a tree in `X`. Kernels
//!   run on `T<&Hbm>`. No tree lives on host: a load stages each tensor on host only until it is
//!   in HBM.

use furiosa_opt_std::prelude::{HbmTensor, M, MaterializableScalar, Scalar};
use furiosa_opt_std::runtime::HbmBuffer as Buffer;
pub use safetensors::Dtype;
use std::io;
use std::ops::RangeBounds;

/// Where every leaf of a tree lives.
pub trait Memory {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static>;
}

/// Device memory.
pub struct Hbm;

impl Memory for Hbm {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = HbmTensor<D, C, S>;
}

impl<'a, X: Memory> Memory for &'a X {
    type Tensor<D: MaterializableScalar, C: M + 'static, S: M + 'static> = &'a X::Tensor<D, C, S>;
}

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
    range: impl RangeBounds<usize>,
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
    use crate::tensor::axis::Eight;
    use crate::tests::DEVICE;
    use furiosa_opt_std::prelude::{Broadcast, Device, Symbol, Topology, m};
    use std::ops::Range;

    /// Storage splits only into whole, nonempty blocks that start 8-byte aligned and span whole
    /// 8-byte words.
    #[test]
    fn refuses_partial_or_unaligned_blocks() -> io::Result<()> {
        let _lease = DEVICE.blocking_lock();
        let device = Device::new(Topology { chips: 1, pes: 8 }).map_err(io::Error::other)?;
        let buffer = Buffer::alloc(&device, 72).map_err(io::Error::other)?;
        let split = |range: Range<usize>, bytes| {
            let buffer = buffer.alias(range).map_err(io::Error::other)?;
            blocks::<HbmTensor<f32, m![1], m![Eight]>>(buffer, bytes)
        };
        assert_eq!(split(0..64, 32)?.len(), 2);
        for (range, bytes) in [(0..72, 32), (4..36, 32), (0..64, 0), (0..72, 36)] {
            let error = split(range, bytes).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "storage does not contain complete aligned blocks"
            );
        }
        Ok(())
    }
}
