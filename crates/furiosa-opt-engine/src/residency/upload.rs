//! Copies prepared weights into HBM, retaining the public runtime's actual transfer completion.
//!
//! On the pinned public SDK, submitting a pinned write hands every piece and its memory leases
//! to the device worker. Its owned future observes all pieces, including queued slots. Retain
//! that future directly; no transfer-slot limit or device-wide synchronization is needed here.

use super::Completion;
use super::source::{Image, Piece, Source};
use furiosa_opt_std::prelude::{Device, HbmBuffer};
use std::{io, sync::Arc};

/// A pageable source is fully copied before placement returns.
pub(super) fn place(device: &mut Device, source: &Source) -> io::Result<HbmBuffer> {
    let pieces = source.pieces(usize::from(device.topology().chips))?;
    let (buffer, work) = write(device, source.len(), pieces)?;
    work.wait().map_err(io::Error::other)?;
    Ok(buffer)
}

/// Issue a pinned image and return its completion separately from its device allocation.
pub(super) fn issue(
    device: &mut Device,
    image: &Image,
) -> io::Result<(HbmBuffer, Option<Arc<Completion>>)> {
    let pieces = image.pieces(usize::from(device.topology().chips))?;
    let (buffer, work) = write(device, image.len(), pieces)?;
    match work.result() {
        Some(Ok(())) => Ok((buffer, None)),
        Some(Err(error)) => Err(io::Error::other(error)),
        None => Ok((buffer, Some(work))),
    }
}

fn write(
    device: &mut Device,
    len: usize,
    pieces: Vec<Piece<'_>>,
) -> io::Result<(HbmBuffer, Arc<Completion>)> {
    let buffer = HbmBuffer::alloc(device, len).map_err(io::Error::other)?;
    let mut aliases = (0..pieces.len())
        .map(|_| buffer.alias(..).map_err(io::Error::other))
        .collect::<io::Result<Vec<_>>>()?;
    let pairs = pieces
        .into_iter()
        .zip(&mut aliases)
        .map(|(piece, to)| {
            let to = to
                .slice_mut(piece.chips, piece.bytes)
                .map_err(io::Error::other)?;
            Ok((piece.host, to))
        })
        .collect::<io::Result<Vec<_>>>()?;
    let work = Arc::new(Completion::new(device.pdma.to_hbm(pairs).submit()));
    Ok((buffer, work))
}

#[cfg(test)]
mod tests {
    use super::*;
    use furiosa_opt_std::prelude::*;

    axes![Out = 8, In = 131072];
    type Chips = m![Out / 2];
    type Shape = m![Out % 2, In];

    /// Pageable and pinned paths preserve both distinct shards and identical chip images. The pinned transfer owns
    /// its source after issue, so freeing the caller's source/image cannot cancel it.
    #[test]
    #[ignore = "requires four explicitly selected RNGD chips and page-locked host memory"]
    fn preserves_prepared_shards_after_the_host_owner_is_dropped() {
        let mut device = Device::new(Topology { chips: 4, pes: 8 }).unwrap();
        for replicated in [false, true] {
            let values: Vec<f32> = (0..Out::SIZE * In::SIZE)
                .map(|at| {
                    let at = if replicated { at % Shape::SIZE } else { at };
                    (at % 251) as f32 - 125.
                })
                .collect();
            let expected: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            let source = Source::from_host::<f32, Chips, Shape, m![Out, In]>(
                HostTensor::from_vec(values),
                "weight",
            )
            .unwrap();
            let pageable = place(&mut device, &source).unwrap();
            let image = source.pin().unwrap();
            drop(source);
            let (pinned, completion) = issue(&mut device, &image).unwrap();
            drop(image);
            let pending = completion.is_some();
            if let Some(completion) = completion {
                completion.wait().unwrap();
            }
            for buffer in [pageable, pinned] {
                let bytes = futures_executor::block_on(
                    buffer
                        .slice(.., ..)
                        .unwrap()
                        .to_host(&mut device.pdma)
                        .submit(),
                )
                .unwrap();
                assert_eq!(&bytes[..], expected);
            }
            println!(
                "prepared upload (replicated={replicated}): {} bytes, async completion returned: {pending}",
                expected.len()
            );
        }
    }
}
