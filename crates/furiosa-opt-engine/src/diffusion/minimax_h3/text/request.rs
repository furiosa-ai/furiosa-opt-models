//! Engine admission budget for the model's checked text request storage.

use super::{Hidden, text};
use furiosa_opt_std::prelude::*;
use std::io;

pub(super) use text::launch::Row;
pub(super) use text::request::Prepared;

/// Reserve the same prompt image, cache, SDK table and hidden-row banks as execution.
pub(super) fn workspace_bytes(device: &mut Device, capacity: usize) -> io::Result<usize> {
    super::super::memory::measure(device, |device| {
        let request = text::request::reserve(device, capacity, capacity)?;
        let table = HbmBuffer::alloc(device, CACHE_TABLE_BYTES).map_err(io::Error::other)?;
        let rows = (0..2 * capacity)
            .map(|_| {
                <Hidden as furiosa_opt_std::runtime::DeviceOutput>::alloc(device)
                    .map(Hidden::into_buffer)
                    .map_err(io::Error::other)
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok((request, table, rows))
    })
}

// The strided Slice address-table header of the SDK revision pinned by Cargo.toml.
const CACHE_TABLE_BYTES: usize = 64;
