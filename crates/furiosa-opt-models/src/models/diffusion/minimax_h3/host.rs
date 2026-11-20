//! Fallible host storage shared by H3 request preparation and media adapters.

use std::{io, mem::size_of};

/// Fill host storage, returning invalid-input errors for unrepresentable byte sizes and
/// out-of-memory errors when its exact reservation fails.
pub fn filled<T: Clone>(count: usize, value: T) -> io::Result<Vec<T>> {
    let bytes = count.checked_mul(size_of::<T>()).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "host buffer byte count overflows usize",
        )
    })?;
    if bytes > isize::MAX as usize {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "host buffer exceeds isize::MAX bytes",
        ));
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| io::Error::new(io::ErrorKind::OutOfMemory, error))?;
    values.resize(count, value);
    Ok(values)
}
